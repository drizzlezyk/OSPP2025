import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TypeVar, Generic
from translation_agent import get_agent_summary

import requests
import yaml

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(asctime)s [%(levelname)s] %(module)s.%(lineno)d %(name)s:\t%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class IssueTrigger:
    trigger_pr_path: str
    issue_title: str
    issue_assignee: str
    file_extension: list[str] = field(default_factory=list)


@dataclass
class Org:
    org_name: str
    issue_of_owner: str
    issue_of_repo: str
    auto_create_issue: bool
    issue_triggers: list[dict | IssueTrigger] = field(default_factory=list)
    change_content_exclude: list[str] = field(default_factory=list)

    def __post_init__(self):
        tmp_issue_triggers: list[IssueTrigger] = []
        for item in self.issue_triggers:
            tmp_issue_triggers.append(IssueTrigger(**item))
        self.issue_triggers = tmp_issue_triggers


@dataclass
class Config:
    orgs: list[dict | Org]

    def __post_init__(self):
        tmp_orgs: list[Org] = []
        for item in self.orgs:
            tmp_orgs.append(Org(**item))
        self.orgs = tmp_orgs


@dataclass
class ReqArgs:
    method: str
    url: str
    headers: dict[str, str]
    params: dict[str, str] | None = field(default=None)
    data: str | None = field(default=None)
    timeout: int = field(default=180)


T = TypeVar('T')
content_type_is_text = "text/plain"
content_type_is_json_dict = {}
content_type_is_json_list = []


def send_request(args: ReqArgs, t: Generic[T]) -> T:
    error_count = 0
    while error_count < 3:
        try:
            resp = requests.request(**args.__dict__)
            resp.raise_for_status()
            if type(t) is dict or type(t) is list:
                res_data: dict | list = resp.json()
            else:
                res_data: str = resp.text
        except requests.exceptions.RequestException as e:
            if e.response.status_code in [400, 401, 403, 404, 405]:
                logger.error("[ERROR] client error {}".format(e))
                break
            logger.error("[ERROR] server error: {}".format(e))
            error_count += 1
        else:
            logger.info("[OK] [{}], {}".format(args.method, args.url))
            return res_data
    return None


class GiteeClient:
    """
    Gitee OpenAPI 客户端
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    def __init__(self, developer_token: str):
        """
        构造函数
        :param developer_token: Gitee v5 token
        """
        self.headers["Authorization"] = "Bearer {}".format(developer_token)

    def get_diff_content(self, owner: str, repo: str, number: int) -> str | None:
        req_url = "https://gitee.com/{}/{}/pulls/{}.diff".format(owner, repo, number)
        req_args = ReqArgs(method="GET", url=req_url, headers=self.headers)
        result: str | None = send_request(req_args, "")
        if result is None:
            logger.error("can not get diff file from PR: {}".format(req_url))
        return result

    def check_issue_exists(self, owner: str, repo: str, issue_titles: list[str]) -> tuple[list[str], list[str]]:
        req_url = "https://gitee.com/api/v5/repos/{}/{}/issues".format(owner, repo)
        page = 1
        existed_issues = []
        while page <= 200:
            query = {
                "per_page": 100,
                "page": page,
                "sort": "created",
                "direction": "desc",
            }
            req_args = ReqArgs(method="GET", url=req_url, params=query, headers=self.headers)
            result: list | None = send_request(req_args, [])
            if result is None:
                break
            page += 1
            for item in result:
                if not issue_titles:
                    return [], existed_issues
                if issue_titles and item.get('title') in issue_titles:
                    issue_titles.remove(item.get('title'))
                    existed_issues.append(item.get('html_url'))
            if len(result) < 100:
                break
        return issue_titles, existed_issues

    def create_issue(self, owner, repo, title, assignee, body):
        req_url = "https://gitee.com/api/v5/repos/{}/issues".format(owner)
        req_body = {
            "repo": repo,
            "title": title,
            "issue_type": "翻译",
            "body": body,
            "assignee": assignee,
            "push_events": False,
            "tag_push_events": False,
            "issues_events": False,
        }
        req_args = ReqArgs(method="POST", url=req_url, headers=self.headers, data=json.dumps(req_body))
        result: dict | None = send_request(req_args, {})
        return result is None

    def add_pr_comment(self, owner, repo, number, body):
        req_url = 'https://gitee.com/api/v5/repos/{}/{}/pulls/{}/comments'.format(owner, repo, number)
        req_body = {
            "body": "### Translation Feedback \n {} ".format(body)
        }
        req_args = ReqArgs(method="POST", url=req_url, headers=self.headers, data=json.dumps(req_body))
        result: dict | None = send_request(req_args, {})
        return result is None

    def check_only_marks_changed(self, owner, repo, number, check_list):
        diff_content = self.get_diff_content(owner, repo, number)
        deleted_strs, inserted_strs = get_diff_content_list(diff_content)
        if is_only_marks_changed(deleted_strs, inserted_strs, check_list):
            logger.warning('Only marks changed, skip the following steps')
            sys.exit(1)
        logger.info('Not just only marks changed, continue creating issue')


def get_diff_file_list(diff_content: str) -> list[str]:
    diff_files_list = []
    diff_files = [x.split(' ')[0][2:] for x in diff_content.split('diff --git ')[1:]]
    for diff_file in diff_files:
        if diff_file.endswith('\"'):
            d = re.compile(r'/[\d\s\S]+')
            diff_file = d.findall(diff_file)
            diff_file = diff_file[0].replace('/', '', 1).replace('\"', '')
            diff_files_list.append(diff_file)
        else:
            diff_files_list.append(diff_file)
    return diff_files_list


def get_diff_content_list(diff_content: str) -> tuple[str, str]:
    pieces = diff_content.split('diff --git')
    deleted_strs = ''
    inserted_strs = ''
    for piece in pieces:
        start = False
        for line in piece.splitlines():
            if line.startswith('@@'):
                start = True
                continue
            if not start:
                continue
            if line.startswith('-'):
                if len(line) == 1:
                    deleted_strs += '\n'
                else:
                    deleted_strs += line[1:]
            elif line.startswith('+'):
                if len(line) == 1:
                    inserted_strs += '\n'
                else:
                    inserted_strs += line[1:]
    return deleted_strs, inserted_strs


def is_only_marks_changed(a, b, check_list):
    s = SequenceMatcher(None, a, b)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'equal':
            continue
        elif tag in ['delete', 'insert']:
            return False
        elif tag == 'replace':
            deleted = ''.join(a[i1:i2]).strip()
            inserted = ''.join(b[j1:j2]).strip()
            if deleted not in check_list or inserted not in check_list:
                return False
    return True


class Args:
    gitee_token: str
    pr_owner: str
    pr_repo: str
    pr_number: int

    def validate(self):
        valid = self.gitee_token and self.pr_owner and self.pr_repo and self.pr_number
        if not valid:
            logger.error("Invalid Command Arguments")
            sys.exit(1)


def load_config_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as config_in:
        data = yaml.safe_load(config_in)

    if data is None:
        return None
    return Config(**data)


def create_issue_based_on_pr_diff_and_config(conf: Config, cli: GiteeClient, pr_owner: str, pr_repo: str,
                                             pr_number: int):
    pr__html_url = "https://gitee.com/{}/{}/pulls/{}".format(pr_owner, pr_repo, pr_number)
    for org_item in conf.orgs:
        issue_title_pr_mark = "{}/{}/pulls/{}".format(pr_owner, pr_repo, pr_number)
        if org_item.org_name != pr_owner:
            continue
        if org_item.auto_create_issue:
            cli.check_only_marks_changed(pr_owner, pr_repo, pr_number, org_item.change_content_exclude)
        file_count = 0
        diff_content = cli.get_diff_content(pr_owner, pr_repo, pr_number)
        if diff_content is None:
            sys.exit(1)
        diff_files = get_diff_file_list(diff_content)
        zh_file = []
        en_file = []
        need_create_issue = {}
        for trigger in org_item.issue_triggers:
            for diff_file in diff_files:
                if diff_file.startswith(trigger.trigger_pr_path) and diff_file.split('.')[-1] in trigger.file_extension:
                    logger.info("file {} has been changed".format(diff_file))
                    file_count += 1
                    if "/zh" in trigger.trigger_pr_path:
                        need_create_issue["zh"] = [trigger.issue_assignee,
                                                   "{}({}).".format(trigger.issue_title, issue_title_pr_mark)]
                        zh_file.append(diff_file.replace("zh/", ""))
                    elif "/en" in trigger.trigger_pr_path:
                        need_create_issue["en"] = [trigger.issue_assignee,
                                                   "{}({}).".format(trigger.issue_title, issue_title_pr_mark)]
                        en_file.append(diff_file.replace("en/", ""))
                    else:
                        logger.warning("not a range")
        changed_same_files = False
        for z in zh_file:
            if z in en_file:
                changed_same_files = True
            else:
                changed_same_files = False
        if file_count == 0:
            logger.warning(
                "NOTE: https://gitee.com/{}/files change files out of translate range".format(issue_title_pr_mark))
            return
        if changed_same_files:
            logger.info("changed the same files in en and zh path, no need to create issue")
            return

        need_create_issue_template = {}
        need_create_issue_titles = []
        for issue_item in need_create_issue:
            need_create_issue_titles.append(need_create_issue[issue_item][1])
            need_create_issue_template[need_create_issue[issue_item][1]] = need_create_issue[issue_item][0]
        if need_create_issue_titles:
            need_create_issue_list, existed_issue_list = cli.check_issue_exists(org_item.issue_of_owner,
                                                                                org_item.issue_of_repo,
                                                                                need_create_issue_titles)
            if not need_create_issue_list:
                feedback_comment = "issue has already created, please go to check issue: {}".format(
                    existed_issue_list)
                logger.info("Warning: " + feedback_comment)
                cli.add_pr_comment(pr_owner, pr_repo, pr_number, feedback_comment)
            for need_create_issue_item in need_create_issue_list:
                
                issue_summary = get_agent_summary(diff_content)
                issue_body = ""
                if issue_summary and not issue_summary.error:
                    issue_body += f"## 📊 变更统计\n\n"
                    issue_body += f"- **总文件数**: {issue_summary.total_files}\n"
                    issue_body += f"- **成功处理文件数**: {issue_summary.processed_files}\n"
                    if issue_summary.total_summary:
                        total = issue_summary.total_summary
                        issue_body += f"- **总改动行数**: {total.total_lines_changed}\n"
                        issue_body += f"- **改动类型**: {', '.join(total.change_type_list)}\n\n"
                        issue_body += f"## 🔍 整体变更摘要\n\n"
                        issue_body += f"{total.overall_summary}\n\n"
                        issue_body += f"## ⚠️ 整体潜在影响\n\n"
                        issue_body += f"{total.overall_potential_impact}\n\n"
                    if issue_summary.file_summaries:
                        issue_body += f"## 📝 单文件变更详情\n\n"
                        for summary in issue_summary.file_summaries:
                            issue_body += f"### 📁 {summary.file_path}\n\n"
                            issue_body += f"- **改动类型**: {summary.change_type}\n"
                            issue_body += f"- **新增行数**: {summary.lines_added}\n"
                            issue_body += f"- **删除行数**: {summary.lines_deleted}\n"
                            issue_body += f"- **潜在影响**: {summary.potential_impact}\n"
                            issue_body += f"- **详细摘要**: {summary.summary}\n\n"
                            issue_body += "---\n\n"
                else:
                    issue_body += f"## ⚠️ 翻译变更检测\n\n"
                    issue_body += f"检测到需要翻译的文件变更，但无法获取详细摘要信息。\n\n"
                    issue_body += f"**变更文件数量**: {len(diff_files)}\n"
                    issue_body += f"**相关PR**: {pr__html_url}\n\n"
                
                issue_body += f"## ❗️ 本Issue的摘要内容基于AI Agent技术自动生成，仅供参考，请以实际更改为准。\n\n" 
                issue_body += f"## 🔗 相关PR链接\n\n"
                issue_body += f"- {pr__html_url}\n"
                
                cli.create_issue(org_item.issue_of_owner, org_item.issue_of_repo, need_create_issue_item,
                                 need_create_issue_template[need_create_issue_item],
                                 issue_body)


def main():
    parser = argparse.ArgumentParser(description='Create Gitee Webhook based on the profile')
    parser.add_argument('--gitee_token', type=str, required=True, help='gitee v5 api token')
    parser.add_argument('--pr_owner', type=str, required=True, help='the PR of owner')
    parser.add_argument('--pr_repo', type=str, required=True, help='the PR of repo')
    parser.add_argument('--pr_number', type=str, required=True, help='the PR number')
    args = Args()
    parser.parse_args(args=sys.argv[1:], namespace=args)
    args.validate()

    exec_py = sys.argv[0]
    config_yaml_path = exec_py[:-2] + 'yaml'
    conf = load_config_yaml(config_yaml_path)

    cli = GiteeClient(args.gitee_token)

    pr_owner = args.pr_owner
    pr_repo = args.pr_repo
    pr_number = args.pr_number
    create_issue_based_on_pr_diff_and_config(conf, cli, pr_owner, pr_repo, pr_number)


if __name__ == '__main__':
    main()
