#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import twitter

CONSUMER_KEY="Tf0llJkFekN0fQRs3jm8D64BV"
CONSUMER_SECRET="iI909hsiksvYfotc9Ra8EPeA8dpCjShB6E2BlmDK31bu1hfLnd"
ACCESS_TOKEN_KEY="952559428619898881-Cimwl8COh51kdsM04xkxNN3mMV8Oy0n"
ACCESS_TOKEN_SECRET="66wC1hUBRI1kO0S5Wvj7vLYhBoKQCIPHQmx0vFFJwuJgw"

USER_NAME='FilipeDeschamps'

api = twitter.Api(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET)

statuses = api.GetUserTimeline(screen_name=USER_NAME)

print(statuses)