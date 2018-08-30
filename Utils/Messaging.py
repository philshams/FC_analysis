from slackclient import SlackClient
import requests
import os

from Config import slack_env_var_token, slack_username

"""
These functions take care of sending slack messages and emails

"""


def slack_chat_messenger(message):
    # NEVER LEAVE THE TOKEN IN YOUR CODE ON GITHUB, EVERYBODY WOULD HAVE ACCESS TO THE CHANNEL!
    slack_token = os.environ.get(slack_env_var_token)
    slack_client = SlackClient(slack_token)

    api_call = slack_client.api_call("im.list")
    user_slack_id = slack_username
    # You should either know the user_slack_id to send a direct msg to the user

    if api_call.get('ok'):
        for im in api_call.get("ims"):
            if im.get("user") == user_slack_id:
                im_channel = im.get("id")
                slack_client.api_call("chat.postMessage", channel=im_channel, text=message, as_user=False)


def slack_chat_attachments(filepath):
    slack_chat_messenger('Trying to send you {}'.format(filepath))

    slack_token = os.environ.get(slack_env_var_token)
    my_file = {
        'file': (filepath+'.png', open(filepath+'.png', 'rb'), 'image/png', {
            'Expires': '0'
        })
    }


    payload = {
      "filename":filepath+'.png',
      "token":slack_token,
      "channels": ['@Fede'],
      "media": my_file
    }

    r = requests.post("https://slack.com/api/files.upload", params=payload)


def upload_file( filepath ):
    """Upload file to channel

    Note:
        URLs can be constructed from:
        https://api.slack.com/methods/files.upload/test
    """
    slack_chat_messenger('Trying to send you {}'.format(filepath))
    slack_token = os.environ.get(slack_env_var_token)

    data = {}
    data['token'] = slack_token
    data['file'] = filepath
    data['filename'] = filepath
    data['channels'] = [slack_username]
    data['display_as_bot'] = True

    filepath = data['file']
    files = {
        'content': (filepath, open(filepath, 'rb'), 'image/png', {
            'Expires': '0'
        })
    }
    data['media'] = files
    response = requests.post(
        url='https://slack.com/api/files.upload',
        data=data,
        headers={'Accept': 'application/json'},
        files=files)

    print(response.text)

def upload(file, channel):
  options = {
    token: @team.bot["bot_access_token"],
    file: File.new("./tmp/composed/#{file.timestamp}", 'rb'),
    filename: "composed_" + file.name,
    title: "Composed " + file.title,
    channels: channel
  }

  res = RestClient.post 'https://slack.com/api/files.upload', options
  json_response = JSON.parse(res.body)

  # Return the uploaded file's ID
  thread_ts = json_response["file"]["shares"]["private"][channel]["ts"]
  file_id = json_response["file"]["id"]
end

