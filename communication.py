import ZipFile
import datetime

import smtplib


https://www.freecodecamp.org/news/send-emails-using-code-4fcea9df63f/


def sendEmil(repo1, repo2, repo3, repo4, repo5, toAdresses, timeVar):
    mail.Subject = "Stock suggestions_" + timeVar
    server = smtplib.SMTP(SERVER)
    server.login("MrDoe", "PASSWORD")
    server.sendmail(FROM, TO, message)
    server.quit()



SERVER = "smtp.example.com"
FROM = "johnDoe@example.com"
TO = ["JaneDoe@example.com"] # must be a list

SUBJECT = "Hello!"
TEXT = "This is a test of emailing through smtp of example.com."

# Prepare actual message
message = """From: %s\r\nTo: %s\r\nSubject: %s\r\n\

%s
""" % (FROM, ", ".join(TO), SUBJECT, TEXT)

