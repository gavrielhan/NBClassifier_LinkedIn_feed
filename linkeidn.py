from linkedin import linkedin



RETURN_URL ='https://www.youtube.com/'
CONSUMER_KEY = '86tdlfi59oh9ke'
CONSUMER_SECRET = '3jLsdGg6mNg6EAGl'

ciao = linkedin.PERMISSIONS.enums
ciao.pop('CONTACT_INFO')
ciao.pop('COMPANY_ADMIN')
ciao.pop('BASIC_PROFILE')
ciao.pop('SHARE')
ciao.pop('EMAIL_ADDRESS')
ciao['FULL_PROFILE'] = 'r_liteprofile'
authentication = linkedin.LinkedInAuthentication(
                    CONSUMER_KEY,
                    CONSUMER_SECRET,
                    RETURN_URL,
                    ciao.values()
                )

print (authentication.authorization_url)

application = linkedin.LinkedInApplication(authentication)

################################################3

from linkedin_api import Linkedin

api = Linkedin('gavriel.hannuna@gmail.com','gh191101')
my_search = api.get_profile('saraspagnoletto')

skills = my_search['skills']

sk=[]
for i in range(len(skills)):
    sk.append(skills[i]['name'])

#############################################

from bot_studio import *
linkedin=bot_studio.linkedin()
linkedin.login(username='gavriel.hannuna@gmail.com', password='gh191101')

def get_post_content(data):
    my_feed=[]
    for dic in data:
        my_feed.append(dic['Post Content'])
        if len(my_feed)>1:
            if my_feed[len(my_feed)-1]=="Show more Feed Updates":
                my_feed.pop(len(my_feed)-1)
            if len(my_feed) > 1:
                if my_feed[len(my_feed)-1]==my_feed[len(my_feed)-2]:
                    my_feed.pop(len(my_feed)-1)
    return my_feed

def find_main_post(my_feed):
    for i in range(len(my_feed)):
        sep_feed = my_feed[i].split("\n")
        len_feed=[]
        for phrase in sep_feed:
            len_feed.append(len(phrase))
        ind = len_feed.index(max(len_feed))
        my_feed[i] = sep_feed[ind]
    return my_feed

def process_feed_data(data):
    my_feed = get_post_content(data)
    my_feed = find_main_post(my_feed)
    return my_feed


data=[]
k=0
while(k<30):
    response=linkedin.get_feed()
    for key in response['body']:
        data.append(key)
    linkedin.scroll()
    k=k+1
    print("Currently at iteration:",k-1)
