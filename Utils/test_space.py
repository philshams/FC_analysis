from slackclient import SlackClient
import requests
import os

# NEVER LEAVE THE TOKEN IN YOUR CODE ON GITHUB, EVERYBODY WOULD HAVE ACCESS TO THE CHANNEL!
slack_token = os.environ.get('SLACK_BRANCO_TOKEN')
slack_client = SlackClient(slack_token)
# ========================================================================================================
# # send attachments
# my_file = {
#   'file' : ('/tmp/myfile.pdf', open('/tmp/myfile.pdf', 'rb'), 'pdf')
# }
# token = ''
# payload={
#   "filename":"myfile.pdf",
#   "token":token,
#   "channels":['#random'],
# }
#
# r = requests.post("https://slack.com/api/files.upload", params=payload, files=my_file)
# # ========================================================================================================
# # To send slack messages to channel
# channel = '#rotation_vte'
# text = 'Test: send message from python'
# slack_client.api_call('chat.postMessage', channel=channel, text=text)

# ========================================================================================================
# To send slack messages to user
api_call = slack_client.api_call("im.list")

# You should either know the user_slack_id to send a direct msg to the user
user_slack_id = "U9ES1UXSM"

if api_call.get('ok'):
    for im in api_call.get("ims"):
        if im.get("user") == user_slack_id:
            im_channel = im.get("id")
            slack_client.api_call("chat.postMessage", channel=im_channel,
                                       text="Test message fgxhfxhxfg python", as_user=False)


# ========================================================================================================
# to send emailsss
# ========================================================================================================

# Import smtplib for the actual sending function
#import smtplib

# Here are the email package modules we'll need
#from email.mime.image import MIMEImage
#from email.mime.text import  MIMEText
# from email.mime.multipart import MIMEMultipart
#
# # Create the container (outer) email message.
# msg = MIMEMultipart()
# msg['Subject'] = 'hiutgo3itgeitg'
# # me == the sender's email address
# # family = the list of all recipients' email addresses
# msg['From'] = 'federicopython@gmail.com'
# msg['To'] = 'federicoclaudi@gmail.com'
# msg.preamble = 'auhfaiuhfiwuehf'
# body = "Python test mail"
# msg.attach(MIMEText(body, 'plain'))
#
# # Assume we know that the image files are all in PNG format
# # for file in pngfiles:
# #     # Open the files in binary mode.  Let the MIMEImage class automatically
# #     # guess the specific image type.
# #     with open(file, 'rb') as fp:
# #         img = MIMEImage(fp.read())
# #     msg.attach(img)
#
# # Send the email via our own SMTP server.
# server = smtplib.SMTP('smtp.gmail.com:587')
# server.ehlo()
# server.starttls()
# server.login('', '')
# server.sendmail('', '', msg.as_string())
# server.quit()