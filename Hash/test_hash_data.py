test_emails = []
test_emails.append("""Dear PhD Students, \nHappy Friday! I wanted to write to send a big thank you to the many students who shared their thoughts through the survey below. \n
\n\nKind regards, \n\nDianne\n""")
test_emails.append("""Hi Cindy,\n\nThanks for the email!
\n1) I’d prefer Tuesday after 5, Thursday after 5:30pm, or anytime on Sunday.
\n\nThank you for your help on organizing our meetings!\n\nBest,\nLara\n----------------------------------------------------
\nLara Yang\nStanford Graduate School of Business\nPhD Student in Organizational Behavior\nlarayang@stanford.edu | 510-604-9056""")
test_emails.append("That works for me.\n\nSent from my iPhone")
test_emails.append("""Hi there, \nLara Yang is inviting you to a scheduled Zoom meeting. \n\nJoin from PC, Mac, Linux, iOS or Android: https://stanford.zoom.us/j/95653872080?pwd=Q0JQRVU1Z09BRkVUVUFHWFN3TTJaQT09\n\tPassword: 880119
\n\nOr iPhone one-tap (US Toll): +18333021536,,95653872080# or +16507249799,,95653872080#\n\nOr Telephone:
\n\nDial: +1 650 724 9799 (US, Canada, Caribbean Toll) or +1 833 302 1536 (US, Canada, Caribbean Toll Free)
\n\n\tMeeting ID: 956 5387 2080\n\tPassword: 880119\n\tInternational numbers available: https://stanford.zoom.us/u/aX6b8tbW9
\n\n\tMeeting ID: 956 5387 2080\n\tPassword: 880119\n\tSIP: 95653872080@zoomcrc.com\n\tPassword: 880119""")
test_emails.append("""Hi Lara,\n \nThanks for your email. Sabby is out sick this week.\nWe will provide you with an update by next week.\n \nThanks,\nAnupama Kar
\nBusiness Analyst – Transformation and GBS\n\n\nT: 1(973) 532-3577   
\nIf you have any questions regarding our pandemic response plan or need additional support please contact us at Crisis_Response@Collabera.com or call  973-841-2220.
\n\nHow am I doing? Give feedback at our Center of Business Excellence or call 1-866-398-6484 
\nPrivacy/Confidentiality""")
test_emails.append("""Hi Lara,\n \nThanks for your email. Sabby is out sick this week.\nWe will provide you with an update by next week.\n \nAnupama Kar
\nBusiness Analyst – Transformation and GBS\n\n\nT: 1(973) 532-3577   
\nIf you have any questions regarding our pandemic response plan or need additional support please contact us at Crisis_Response@Collabera.com or call  973-841-2220.
\n\nHow am I doing? Give feedback at our Center of Business Excellence or call 1-866-398-6484 
\nPrivacy/Confidentiality""")
test_emails.append("")
test_emails.append("---------------------------------")
test_emails.append("Ability to work well in an independent setting with minimal supervisionAbility to work collaboratively within a teamAbility to communicate effectively with patients or their respective representativesStrong customer service skillsStrong individual accountability and ownership .")
test_emails.append("""CAUTION : - External Mail .
PPPPP PPPPP P PPPPP PPPPPPPP actively protects your mailbox by quarantining spam and other unwanted email .
Use this digest to manage quarantined messages and approve sender addresses .
Important : Do NOT forward this message .
Recipients of this message will be able to manage your quarantined messages and approve senders .
For more information about this digest , contact your mail administrator .
Other Ways to Manage Quarantined Messages 
 
 The following summary displays a maximum of 000 of the most recent quarantined spam messages , if you need to manage your all quarantined messages , please log on the End User Console at : 
  Summary 
  Your email address : 
  Digest date : 00 / 00 / 0000 00 : 00 : 00 
  New found messages in quarantine : 0 of 0 
  Quarantine Digest 
  Quarantined 
  Sender 
  Recipient 
  Subject 
  Manage Messages 
  00 / 00 / 0000 00 : 00 : 00 
  nidhi . ajoshi @ collabera . com 
  Introducing Dataiku 0 : Now with Fully Managed Kubernetes for Elasticity and Collaboration""")
test_emails.append("Get it done asap. \nThx, \nLara")
test_emails.append("Get it done asap. \nbegin forwarded message:\n how's life?")
test_emails.append("NNNNN NNNNN 00 : 00 AM : Hey ! ! ! !\n!")
test_emails.append("""NNNNN NNNNN 0 : 00 PM : NNNN N N NNNNNN NNNNN 0 : 00 PM : Haan bhai . . . let's go . . . 
 
 chalte hai .
NNNNN NNNNN 0 : 00 PM : help chahiye 
 
 confidential hai 
 
 NNNNNN NNNNN 0 : 00 PM : Haan bol na . . . 
 
 Are bol to . . 
 
 NNNNN NNNNN 0 : 00 PM : apney chaltey hai fir baat krty 
 
 NNNNNN NNNNN 0 : 00 PM : Yap sure . . . chal le . . . 
 
 NNNNN NNNNN 0 : 00 PM : hmmmm 
 
 NNNNN NNNNN 00 : 00 AM : chale bhaiya ji 
 
 NNNNNN NNNNN 00 : 00 AM : Haanji . . . give me 0 to 00 mins pls .
NNNNN NNNNN 00 : 00 AM : okk ji 
 
 NNNNNN NNNNN 00 : 00 AM : chalo bhai . . 
 
 chalte hai 
 
 NNNNN NNNNN 00 : 00 AM : chaliye fir 
 
 NNNNNN NNNNN 00 : 00 AM : NNNN NNNNN pi ne chalta kya ?""")
 

correct_clean_body = ["dear phd students\nhappy friday SENT_END i wanted to write to send a big thank you to the many students who shared their thoughts through the survey below SENT_END\n",
"hi cindy\nthanks for the email SENT_END\n1 i would prefer tuesday after 5 thursday after pm or anytime on sunday SENT_END\nthank you for your help on organizing our meetings SENT_END\n",
"that works for me SENT_END\n",
"",
"hi lara\nthanks for your email SENT_END sabby is out sick this week SENT_END\nwe will provide you with an update by next week SENT_END\n",
"hi lara\nthanks for your email SENT_END sabby is out sick this week SENT_END\nwe will provide you with an update by next week SENT_END\nanupama kar\nbusiness analyst transformation and gbs\nt\n",
"",
"",
"ability to work well in an independent setting with minimal supervision ability to work collaboratively within a team ability to communicate effectively with patients or their respective representatives strong customer service skills strong individual accountability and ownership SENT_END\n",
"",
"get it done asap SENT_END\n",
"get it done asap SENT_END\n",
"",
""]


correct_hashed_body = ["ff17bacf 4295ed0c 75d37c6c\n56ab24c1 f6f7fec0 SENT_END 865c0c0b dc1f3d93 01b6e203 efb2a684 01b6e203 2541d938 0cc175b9 d861877d f5ab9692 639bae9a 01b6e203 8fc42c6d 8cd283d8 75d37c6c 53d670af 9e81e7b9 0e66be14 eea6456d ca23ba20 8fc42c6d 2fa7b041 6cede1cf SENT_END\n",
"""49f68a5c cc4b2066\n71d3e8b4 d5566982 8fc42c6d 0c83f57c SENT_END\nNUM 865c0c0b e680afd3 02a04869 1a31a6f6 632a2406 NUM c395246f 632a2406 5109d85d e81c4e4f 2f1002f0 ed2b5c01 787c74a2 SENT_END\nf5ab9692 639bae9a d5566982 62cc0b4e 657f8b8d ed2b5c01 8d051c1c 162e31af 06cf60d2 SENT_END\n""",
"21582c6c 038703c7 d5566982 ab86a1e1 SENT_END\n",
"",
"""49f68a5c d3c327c8\n71d3e8b4 d5566982 62cc0b4e 0c83f57c SENT_END aecdc7bb a2a551a6 c68271a6 8d7d5ffd 9e925e93 172a8327 SENT_END\nff1ccf57 18218139 a4f550de 639bae9a 23a58bf9 18b049cc 3ac34083 df3f079d d0cab90d 172a8327 SENT_END\n""",
"""49f68a5c d3c327c8\n71d3e8b4 d5566982 62cc0b4e 0c83f57c SENT_END aecdc7bb a2a551a6 c68271a6 8d7d5ffd 9e925e93 172a8327 SENT_END\nff1ccf57 18218139 a4f550de 639bae9a 23a58bf9 18b049cc 3ac34083 df3f079d d0cab90d 172a8327 SENT_END\nf4a46706 aa8ae3b3\nf5d7e253 05d5c5df 3935f8fe be5d5d37 c7c16ad0\ne358efa4\n""",
"",
"",
"424dbe53 01b6e203 67e92c87 f9323f5b 13b5bfe9 18b049cc e2580777 7dc22b2c 23a58bf9 dc43e863 13b789b8 424dbe53 01b6e203 67e92c87 284faeb1 60df9e64 0cc175b9 f894427c 424dbe53 01b6e203 01b922d8 c4842cc4 23a58bf9 3495d5d8 e81c4e4f 0e66be14 3741740c 19494e3c 6f7f9432 91ec1f93 aaabf0d3 a658279f 6f7f9432 23b79ae0 dd2d6814 be5d5d37 b46b517b SENT_END\n",
"",
"b5eda0a7 0d149b90 6b2ded51 3d630eac SENT_END\n",
"b5eda0a7 0d149b90 6b2ded51 3d630eac SENT_END\n",
"",
""]


multi_lang_emails = test_emails + ["bhai meko kya malum ki 0000 me paas ho re sab ne salo ne 0000 likha hua tha \nsahi kaha\n",
"bonjour mon chéri,\nnous nous tenons à votre disposition pour toute question.",
"中文的邮件处理的了吗？"]


correct_lang = [True, True, True, False, True, True, False, False, True, False, True, True, False, False, False, False, False]

correct_liwc = [{'Funct': 13, 'Social': 6, 'Affect': 5, 'Posemo': 5, 'Prep': 5, 'Relativ': 4, 'Pronoun': 4, 'Work': 3, 'Ppron': 3, 'Verbs': 3, 'CogMech': 3, 'Article': 3, 'Past': 2, 'Space': 2, 'Time': 1, 'I': 1, 'Discrep': 1, 'Motion': 1, 'Present': 1, 'You': 1, 'Quant': 1, 'Ipron': 1, 'They': 1, 'Insight': 1},
{'Funct': 13, 'Relativ': 8, 'Social': 7, 'Prep': 6, 'Time': 6, 'CogMech': 4, 'Verbs': 3, 'Affect': 3, 'Posemo': 3, 'Ppron': 4, 'Pronoun': 4, 'Present': 2, 'Discrep': 2, 'Tentat': 2, 'Space': 2, 'Article': 1, 'I': 1, 'AuxVb': 1, 'Future': 1, 'Insight': 1, 'Conj': 1, 'Excl': 1, 'You': 2, 'We': 1, 'Work': 1},
{'Funct': 3, 'Pronoun': 2, 'Ipron': 1, 'Work': 1, 'Achiev': 1, 'Prep': 1, 'Ppron': 1, 'I': 1},
{},
{'Funct': 11, 'Social': 6, 'Relativ': 5, 'Prep': 4, 'Time': 4, 'Verbs': 3, 'CogMech': 3, 'Incl': 3, 'Pronoun': 4, 'Present': 2, 'AuxVb': 2, 'Ppron': 3, 'Affect': 1, 'Posemo': 1, 'Space': 1, 'Bio': 1, 'Health': 1, 'Ipron': 1, 'We': 1, 'Future': 1, 'You': 2, 'Article': 1},
{'Funct': 12, 'Social': 6, 'Relativ': 5, 'Prep': 4, 'CogMech': 4, 'Incl': 4, 'Time': 4, 'Verbs': 3, 'Pronoun': 4, 'Present': 2, 'AuxVb': 2, 'Ppron': 3, 'Affect': 1, 'Posemo': 1, 'Space': 1, 'Bio': 1, 'Health': 1, 'Ipron': 1, 'We': 1, 'Future': 1, 'You': 2, 'Article': 1, 'Work': 1, 'Money': 1, 'Conj': 1},
{},
{},
{'Funct': 13, 'Prep': 7, 'Achiev': 7, 'CogMech': 6, 'Work': 5, 'Social': 4, 'Affect': 3, 'Posemo': 3, 'Incl': 3, 'Relativ': 2, 'Space': 2, 'Article': 2, 'Cause': 2, 'Conj': 2, 'Money': 2, 'Nonflu': 1, 'Adverbs': 1, 'Leisure': 1, 'Insight': 1, 'Tentat': 1, 'Excl': 1, 'They': 1, 'Ppron': 1, 'Pronoun': 1, 'Humans': 1},
{},
{'Verbs': 2, 'Funct': 2, 'Present': 1, 'Ipron': 1, 'Pronoun': 1, 'Past': 1, 'AuxVb': 1},
{'Verbs': 2, 'Funct': 2, 'Present': 1, 'Ipron': 1, 'Pronoun': 1, 'Past': 1, 'AuxVb': 1},
{},
{}]