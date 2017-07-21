#!/usr/bin/env python3

import argparse
import base64
import hashlib
import json
import os
import requests
import sys
import time

def sleep_time(attempt):
    if attempt <= 0:
        raise Exception('Unexpected')
    if attempt == 1:
        return 0
    if attempt == 2:
        return 15
    if attempt == 3:
        return 60
    if attempt == 4:
        return 90
    if attempt == 5:
        return 300
    return 1200

def retry(func_in):
    def func_out(*args, **kwargs):
        retry_max = 10
        i = 0
        while True:
            i = i + 1
            try:
                return func_in(*args, **kwargs)
            except Exception as exc:
                if i > retry_max:
                    raise exc
                print('Operation failed. Exception:\n  {}'.format(exc))
                sec = sleep_time(i)
                print('Retry #{} (of {}) after {} seconds'.format(i, retry_max, sec))
                time.sleep(sec)
        raise Exception('Unreachable')
    return func_out

# http://stackoverflow.com/a/16696317/2288008
@retry
def download_file(url, local_file, auth, chunk_size=1024):
    print('Downloading:\n  {}\n  -> {}'.format(url, local_file))
    r = requests.get(url, stream=True, auth=auth)
    if not r.ok:
        raise Exception('Downloading failed')
    with open(local_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)

class Github:
    def __init__(self, username, password, repo_owner, repo):
        self.repo_owner = repo_owner
        self.repo = repo
        self.auth = requests.auth.HTTPBasicAuth(username, password)
        self.simple_request()

    @retry
    def simple_request(self):
        print('Processing simple request')
        r = requests.get('https://api.github.com', auth=self.auth)
        if not r.ok:
            sys.exit('Simple request fails. Check your password.')

        limit = int(r.headers['X-RateLimit-Remaining'])
        print('GitHub Limit: {}'.format(limit))
        if limit == 0:
            raise Exception('GitHub limit is 0')
        print('Simple request pass')

    @retry
    def get_release_by_tag(self, tagname):
        print('Get release-id by tag `{}`'.format(tagname))
        # https://developer.github.com/v3/repos/releases/#get-a-release-by-tag-name
        # GET /repos/:owner/:repo/releases/tags/:tag

        url = 'https://api.github.com/repos/{}/{}/releases/tags/{}'.format(
            self.repo_owner,
            self.repo,
            tagname
        )

        r = requests.get(url, auth=self.auth)
        if not r.ok:
            raise Exception('Get tag id failed. Requested url: {}'.format(url))

        tag_id = r.json()['id']
        print('Tag id is {}'.format(tag_id))
        return tag_id

    @retry
    def find_asset_id_by_name(self, release_id, name):
        # https://developer.github.com/v3/repos/releases/#list-assets-for-a-release
        # GET /repos/:owner/:repo/releases/:id/assets

        page_number = 1
        keep_searching = True

        while keep_searching:
            url = 'https://api.github.com/repos/{}/{}/releases/{}/assets?page={}'.format(
                self.repo_owner,
                self.repo,
                release_id,
                page_number
            )

            print('Requesting URL: {}'.format(url))
            r = requests.get(url, auth=self.auth)
            if not r.ok:
                raise Exception('Getting list of assets failed. Requested url: {}'.format(url))

            json = r.json()

            for x in json:
                if name == x['name']:
                    return x['id']

            if not json:
                keep_searching = False

            page_number = page_number + 1

        return None

    @retry
    def delete_asset_by_id(self, asset_id, asset_name):
        # https://developer.github.com/v3/repos/releases/#delete-a-release-asset
        # DELETE /repos/:owner/:repo/releases/assets/:id

        url = 'https://api.github.com/repos/{}/{}/releases/assets/{}'.format(
            self.repo_owner,
            self.repo,
            asset_id
        )

        r = requests.delete(url, auth=self.auth)
        if r.status_code == 204:
            print('Asset removed: {}'.format(asset_name))
        else:
            raise Exception('Deletion of asset failed: {}'.format(asset_name))

    def delete_asset_if_exists(self, release_id, asset_name):
        asset_id = self.find_asset_id_by_name(release_id, asset_name)
        if not asset_id:
            print('Asset not exists: {}'.format(asset_name))
            return
        self.delete_asset_by_id(asset_id, asset_name)

    def upload_bzip_once(self, url, local_path):
        headers = {'Content-Type': 'application/x-bzip2'}
        file_to_upload = open(local_path, 'rb')
        r = requests.post(url, data=file_to_upload, headers=headers, auth=self.auth)
        if not r.ok:
            raise Exception('Upload of file failed')

    @retry
    def upload_bzip(self, url, local_path, release_id, asset_name):
        print('Uploading:\n  {}\n  -> {}'.format(local_path, url))
        try:
            self.upload_bzip_once(url, local_path)
        except Exception as exc:
            print('Exception catched while uploading, removing asset...')
            self.delete_asset_if_exists(release_id, asset_name)
            raise exc

    def upload_raw_file(self, local_path):
        tagname = 'cache'
        release_id = self.get_release_by_tag(tagname)

        # https://developer.github.com/v3/repos/releases/#upload-a-release-asset
        # POST https://<upload_url>/repos/:owner/:repo/releases/:id/assets?name=foo.zip

        asset_name = hashlib.sha1(open(local_path, 'rb').read()).hexdigest()
        asset_name = asset_name + '.tar.bz2'

        url = 'https://uploads.github.com/repos/{}/{}/releases/{}/assets?name={}'.format(
            self.repo_owner,
            self.repo,
            release_id,
            asset_name
        )

        self.upload_bzip(url, local_path, release_id, asset_name)

    @retry
    def create_new_file(self, local_path, github_path):
        # https://developer.github.com/v3/repos/contents/#create-a-file
        # PUT /repos/:owner/:repo/contents/:path

        message = 'Uploading cache info\n\n'
        message += 'Create file: {}\n\n'.format(github_path)

        env_list = []
        job_url = ''

        if os.getenv('TRAVIS') == 'true':
            # * https://docs.travis-ci.com/user/environment-variables/#Default-Environment-Variables
            message += 'Travis:\n'
            job_url = 'https://travis-ci.org/{}/jobs/{}'.format(
                os.getenv('TRAVIS_REPO_SLUG'),
                os.getenv('TRAVIS_JOB_ID')
            )

            env_list += [
                'TRAVIS_BRANCH',
                'TRAVIS_BUILD_ID',
                'TRAVIS_BUILD_NUMBER',
                'TRAVIS_JOB_ID',
                'TRAVIS_JOB_NUMBER',
                'TRAVIS_OS_NAME',
                'TRAVIS_REPO_SLUG'
            ]

        if os.getenv('APPVEYOR') == 'True':
            # * http://www.appveyor.com/docs/environment-variables
            message += 'AppVeyor:\n'
            job_url = 'https://ci.appveyor.com/project/{}/{}/build/{}/job/{}'.format(
                os.getenv('APPVEYOR_ACCOUNT_NAME'),
                os.getenv('APPVEYOR_PROJECT_SLUG'),
                os.getenv('APPVEYOR_BUILD_VERSION'),
                os.getenv('APPVEYOR_JOB_ID')
            )
            env_list += [
                'APPVEYOR_ACCOUNT_NAME',
                'APPVEYOR_PROJECT_ID',
                'APPVEYOR_PROJECT_NAME',
                'APPVEYOR_PROJECT_SLUG',
                'APPVEYOR_BUILD_ID',
                'APPVEYOR_BUILD_NUMBER',
                'APPVEYOR_BUILD_VERSION',
                'APPVEYOR_JOB_ID',
                'APPVEYOR_JOB_NAME',
                'APPVEYOR_REPO_BRANCH'
            ]

        # Store some info about build
        for env_name in env_list:
            env_value = os.getenv(env_name)
            if env_value:
                message += '  {}: {}\n'.format(env_name, env_value)

        if job_url:
            message += '\n  Job URL: {}\n'.format(job_url)

        url = 'https://api.github.com/repos/{}/{}/contents/{}'.format(
            self.repo_owner,
            self.repo,
            github_path
        )

        content = base64.b64encode(open(local_path, 'rb').read()).decode()

        put_data = {
            'message': message,
            'content': content
        }

        r = requests.put(url, data = json.dumps(put_data), auth=self.auth)
        if not r.ok:
            print('Put failed. Status code: {}'.format(r.status_code))
            if r.status_code == 409:
                raise Exception('Unavailable repository')
        return r.ok

class CacheEntry:
    def __init__(self, cache_done_path, cache_dir, temp_dir):
        self.cache_dir = cache_dir
        self.temp_dir = temp_dir
        self.cache_raw = os.path.join(self.cache_dir, 'raw')
        self.cache_meta = os.path.join(self.cache_dir, 'meta')
        self.cache_done_path = cache_done_path
        if not os.path.exists(cache_done_path):
            raise Exception('File not exists: {}'.format(cache_done_path))
        self.cache_done_dir = os.path.dirname(self.cache_done_path)
        self.from_server = os.path.join(self.cache_done_dir, 'from.server')
        self.cache_sha1 = os.path.join(self.cache_done_dir, 'cache.sha1')

        self.internal_deps_id = os.path.split(self.cache_done_dir)[0]
        self.type_id = os.path.split(self.internal_deps_id)[0]
        self.args_id = os.path.split(self.type_id)[0]
        self.archive_id = os.path.split(self.args_id)[0]
        self.version = os.path.split(self.archive_id)[0]
        self.component = os.path.split(self.version)[0]
        if os.path.split(self.component)[1].startswith('__'):
            self.package = os.path.split(self.component)[0]
        else:
            self.package = self.component
            self.component = ''
        self.toolchain_id = os.path.split(self.package)[0]
        meta = os.path.split(self.toolchain_id)[0]
        assert(meta == self.cache_meta)

    def entry_from_server(self):
        return os.path.exists(self.from_server)

    def upload_raw(self, github):
        sha1 = open(self.cache_sha1, 'r').read()
        raw = os.path.join(self.cache_raw, sha1 + '.tar.bz2')
        github.upload_raw_file(raw)

    def upload_meta(self, github, cache_done):
        self.upload_files_from_common_dir(github, self.cache_done_dir, cache_done)
        self.upload_files_from_common_dir(github, self.internal_deps_id, cache_done)
        self.upload_files_from_common_dir(github, self.type_id, cache_done)
        self.upload_files_from_common_dir(github, self.args_id, cache_done)
        self.upload_files_from_common_dir(github, self.archive_id, cache_done)
        self.upload_files_from_common_dir(github, self.version, cache_done, check_is_empty=True)
        if self.component != '':
            self.upload_files_from_common_dir(github, self.component, cache_done, check_is_empty=True)
        self.upload_files_from_common_dir(github, self.package, cache_done, check_is_empty=True)
        self.upload_files_from_common_dir(github, self.toolchain_id, cache_done)

    def upload_files_from_common_dir(self, github, dir_path, cache_done, check_is_empty=False):
        to_upload = []
        for i in os.listdir(dir_path):
            if i == 'cmake.lock':
                continue
            if i == 'DONE':
                continue
            done_file = (i == 'CACHE.DONE') or (i == 'basic-deps.DONE')
            if done_file and not cache_done:
                continue
            if not done_file and cache_done:
                continue
            i_fullpath = os.path.join(dir_path, i)
            if os.path.isfile(i_fullpath):
                to_upload.append(i_fullpath)
        if not cache_done:
            if check_is_empty and len(to_upload) != 0:
                raise Exception('Expected no files in directory: {}'.format(dir_path))
            if not check_is_empty and len(to_upload) == 0:
                raise Exception('No files found in directory: {}'.format(dir_path))
        for i in to_upload:
            relative_path = i[len(self.cache_meta)+1:]
            relative_unix_path = relative_path.replace('\\', '/') # convert windows path
            expected_download_url = 'https://raw.githubusercontent.com/{}/{}/master/{}'.format(
                github.repo_owner,
                github.repo,
                relative_unix_path
            )
            github_url = 'https://github.com/{}/{}/blob/master/{}'.format(
                github.repo_owner,
                github.repo,
                relative_unix_path
            )
            print('Uploading file: {}'.format(relative_path))
            ok = github.create_new_file(i, relative_unix_path)
            if not ok:
                print('Already exist')
                temp_file = os.path.join(self.temp_dir, '__TEMP.FILE')
                download_file(expected_download_url, temp_file, github.auth)
                expected_content = open(i, 'rb').read()
                downloaded_content = open(temp_file, 'rb').read()
                expected_hash = hashlib.sha1(expected_content).hexdigest()
                downloaded_hash = hashlib.sha1(downloaded_content).hexdigest()
                os.remove(temp_file)
                if expected_hash != downloaded_hash:
                    print('Hash mismatch:')
                    print(
                        '  expected {} (content: {})'.format(
                            expected_hash, expected_content
                        )
                    )
                    print(
                        '  downloaded {} (content: {})'.format(
                            downloaded_hash, downloaded_content
                        )
                    )
                    print('GitHub link: {}'.format(github_url))
                    raise Exception('Hash mismatch')

class Cache:
    def __init__(self, cache_dir, temp_dir):
        self.entries = self.create_entries(cache_dir, temp_dir)
        self.remove_entries_from_server()
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

    def create_entries(self, cache_dir, temp_dir):
        print('Searching for CACHE.DONE files in directory:\n  {}\n'.format(cache_dir))
        entries = []
        for root, dirs, files in os.walk(cache_dir):
            for filename in files:
                if filename == 'CACHE.DONE':
                    entries.append(CacheEntry(os.path.join(root, filename), cache_dir, temp_dir))
        print('Found {} files:'.format(len(entries)))
        for i in entries:
            print('  {}'.format(i.cache_done_path))
        print('')
        return entries

    def remove_entries_from_server(self):
        new_entries = []
        for i in self.entries:
            if i.entry_from_server():
                print('Remove entry (from server):\n  {}'.format(i.cache_done_path))
            else:
                new_entries.append(i)
        self.entries = new_entries

    def upload_raw(self, github):
        for i in self.entries:
            i.upload_raw(github)

    def upload_meta(self, github, cache_done):
        for i in self.entries:
            i.upload_meta(github, cache_done)

parser = argparse.ArgumentParser(
    description='Script for uploading Hunter cache files to GitHub'
)

parser.add_argument(
    '--username',
    required=True,
    help='Username'
)

parser.add_argument(
    '--repo-owner',
    required=True,
    help='Repository owner'
)

parser.add_argument(
    '--repo',
    required=True,
    help='Repository name'
)

parser.add_argument(
    '--cache-dir',
    required=True,
    help='Hunter cache directory, e.g. /home/user/.hunter/_Base/Cache'
)

parser.add_argument(
    '--temp-dir',
    required=True,
    help='Temporary directory where files will be downloaded for verification'
)

parser.add_argument(
    '--skip-raw', action='store_true', help="Skip uploading of raw files"
)

args = parser.parse_args()

cache_dir = os.path.normpath(args.cache_dir)

# Some tests don't produce cache for some toolchains:
# * https://travis-ci.org/ingenue/hunter/jobs/185550289
if not os.path.exists(cache_dir):
    print("*** WARNING *** Cache directory '{}' not found, skipping...".format(cache_dir))
    sys.exit()

if not os.path.isdir(cache_dir):
    raise Exception('Not a directory: {}'.format(cache_dir))

if os.path.split(cache_dir)[1] != 'Cache':
    raise Exception('Cache directory path should ends with Cache: {}'.format(cache_dir))

cache = Cache(cache_dir, args.temp_dir)

password = os.getenv('GITHUB_USER_PASSWORD')

if password == '' or password is None:
    raise Exception('Expected GITHUB_USER_PASSWORD environment variable')

github = Github(
    username = args.username,
    password = password,
    repo_owner = args.repo_owner,
    repo = args.repo
)

if args.skip_raw:
    print('*** WARNING *** Skip uploading of raw files')
else:
    cache.upload_raw(github)

cache.upload_meta(github, cache_done=False)
print('Uploading DONE files')
cache.upload_meta(github, cache_done=True) # Should be last
