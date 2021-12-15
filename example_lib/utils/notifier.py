from typing import List
import os
import datetime
import traceback
import functools
import json
import socket
import requests
import time
import hmac
import hashlib
import base64
import urllib

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def dingtalk_sender(webhook_url: str,
                           user_mentions: List[str] = [],
                           secret: str = '',
                           proxy_addr: str = None,
                           keywords: List[str] = []):
    """
    DingTalk sender wrapper: execute func, send a DingTalk notification with the end status
    (sucessfully finished or crashed) at the end. Also send a DingTalk notification before
    executing func.
    `webhook_url`: str
        The webhook URL to access your DingTalk chatroom.
        Visit https://ding-doc.dingtalk.com/doc#/serverapi2/qf2nxq for more details.
    `user_mentions`: List[str] (default=[])
        Optional users phone number to notify.
        Visit https://ding-doc.dingtalk.com/doc#/serverapi2/qf2nxq for more details.
    `secret`: str (default='')
        DingTalk chatroom robot are set with at least one of those three security methods
        (ip / keyword / secret), the chatroom will only accect messages that:
            are from authorized ips set by user (ip),
            contain any keyword set by user (keyword),
            are posted through a encrypting way (secret).
        Vist https://ding-doc.dingtalk.com/doc#/serverapi2/qf2nxq from more details.
    `keywords`: List[str] (default=[])
        see `secret`
    """
    msg_template_md = {
        "msgtype": "markdown",
        "markdown": {
            "title": "# Multisource: Art -> Clipart",
            "text": "",
        },
        "at": {
            "atMobiles": user_mentions,
            "isAtAll": False
        }
    }
    timeout_thresh = 5

    def _construct_encrypted_url():
        '''
        Visit https://ding-doc.dingtalk.com/doc#/serverapi2/qf2nxq for details
        '''
        timestamp = round(datetime.datetime.now().timestamp() * 1000)
        secret_enc = secret.encode('utf-8')
        string_to_sign = '{}\n{}'.format(timestamp, secret)
        string_to_sign_enc = string_to_sign.encode('utf-8')
        hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        encrypted_url = webhook_url + '&timestamp={}'.format(timestamp) \
                        + '&sign={}'.format(sign)
        return encrypted_url

    def robust_post(url, msg):
        if proxy_addr is not None:
            proxies = {
                "http": f"{proxy_addr}",
                "https": f"{proxy_addr}"
            }
        try:
            requests.post(url, json=msg, timeout=timeout_thresh)
        except requests.exceptions.Timeout:
            requests.post(url, json=msg, timeout=timeout_thresh, proxies=proxies)
        except Exception as e:
            print("Post Failed {}: ".format(e), msg)

    def decorator_sender(func):
        @functools.wraps(func)
        def wrapper_sender(*args, **kwargs):

            start_time = datetime.datetime.now()
            host_name = socket.gethostname()
            func_name = func.__name__

            # Handling distributed training edge case.
            # In PyTorch, the launch of `torch.distributed.launch` sets up a RANK environment variable for each process.
            # This can be used to detect the master process.
            # See https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py#L211
            # Except for errors, only the master process will send notifications.
            if 'RANK' in os.environ:
                master_process = (int(os.environ['RANK']) == 0)
                host_name += ' - RANK: %s' % os.environ['RANK']
            else:
                master_process = True

            if master_process:
                contents = ['# Your training has started 🎬\n',
                            '- Machine name: %s' % host_name,
                            '- Main call: %s' % func_name,
                            '- Starting date: %s' % start_time.strftime(DATE_FORMAT)]
                contents.extend(['@{}'.format(i) for i in user_mentions])
                if len(keywords):
                    contents.append('\nKeywords: {}'.format(', '.join(keywords)))

                msg_template_md['markdown']['title'] = 'Your training has started 🎬'
                msg_template_md['markdown']['text'] = '\n'.join(contents)
                if secret:
                    postto = _construct_encrypted_url()
                    robust_post(postto, msg=msg_template_md)
                else:
                    robust_post(webhook_url, msg=msg_template_md)

            try:
                value = func(*args, **kwargs)

                if master_process:
                    end_time = datetime.datetime.now()
                    elapsed_time = end_time - start_time
                    contents = ['# Your training is complete 🎉\n',
                                ' - Machine name: %s' % host_name,
                                ' - Main call: %s' % func_name,
                                ' - Starting date: %s' % start_time.strftime(DATE_FORMAT),
                                ' - End date: %s' % end_time.strftime(DATE_FORMAT),
                                ' - Training duration: %s' % str(elapsed_time)]

                    try:
                        task_name = value['task']
                        task_name = task_name.replace('/', '+')
                        task_name = task_name.replace('->', '➡︎')
                        contents.append('\nMain call returned value:\n')
                        contents.append('## {}\n'.format(task_name))
                        contents.append('> **Best**')
                        for dn in value['best']:
                            contents.append('> - Accuracy@{}: {:.5f}'.format(dn, value['best'][dn]))
                        contents.append('>\n> **Final**')
                        for dn in value['final']:
                            contents.append('> - Accuracy@{}: {:.5f}'.format(dn, value['final'][dn]))
                    except:
                        contents.append('Main call returned value: %s' % "ERROR - Couldn't str the returned value.")

                    contents.extend(['@{}'.format(i) for i in user_mentions])
                    if len(keywords):
                        contents.append('\nKeywords: {}'.format(', '.join(keywords)))

                    msg_template_md['markdown']['title'] = 'Your training is complete 🎉'
                    msg_template_md['markdown']['text'] = '\n'.join(contents)
                    if secret:
                        postto = _construct_encrypted_url()
                        robust_post(postto, msg=msg_template_md)
                    else:
                        robust_post(webhook_url, msg=msg_template_md)
                        print(msg_template_md)

                return value

            except Exception as ex:
                end_time = datetime.datetime.now()
                elapsed_time = end_time - start_time
                contents = ["# Your training has crashed ☠️\n",
                            '- Machine name: %s' % host_name,
                            '- Main call: %s' % func_name,
                            '- Starting date: %s' % start_time.strftime(DATE_FORMAT),
                            '- Crash date: %s' % end_time.strftime(DATE_FORMAT),
                            '- Crashed training duration: %s\n\n' % str(elapsed_time),
                            "## Here's the error:\n",
                            '%s\n\n' % ex,
                            "> Traceback:",
                            '> %s' % traceback.format_exc()]
                contents.extend(['@{}'.format(i) for i in user_mentions])
                if len(keywords):
                    contents.append('\nKeywords: {}'.format(', '.join(keywords)))

                msg_template_md['markdown']['text'] = '\n'.join(contents)
                if secret:
                    postto = _construct_encrypted_url()
                    robust_post(postto, msg=msg_template_md)
                else:
                    robust_post(webhook_url, msg=msg_template_md)
                    print(msg_template_md)

                raise ex

        return wrapper_sender

    return decorator_sender
