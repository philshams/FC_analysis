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

    r = requests.post("https://slack.com/api/files.upload", params=payload, files=my_file)
    print(r.text)

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


def send_email_attachments(filename, filepath):
    import smtplib
    from email.mime.image import MIMEImage
    from email.mime.text import  MIMEText
    from email.mime.multipart import MIMEMultipart

    # Create the container (outer) email message.
    msg = MIMEMultipart()
    msg['Subject'] = filename
    # me == the sender's email address
    # family = the list of all recipients' email addresses
    msg['From'] = 'federicopython@gmail.com'
    msg['To'] = 'federicoclaudi@gmail.com'
    body = "Analysis results"
    msg.attach(MIMEText(body, 'plain'))


    with open(filepath+'.png', 'rb') as fp:
        img = MIMEImage(fp.read())
    msg.attach(img)

    # Send the email via our own SMTP server.
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.ehlo()
    server.starttls()
    server.login('federicopython@gmail.com', '')
    server.sendmail('federicopython@gmail.com', 'federicoclaudi@gmail.com', msg.as_string())
    server.quit()

