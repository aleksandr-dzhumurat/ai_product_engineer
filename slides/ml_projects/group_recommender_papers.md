# Group recommender system

The goal is to develop a system that helps to decide how to spend time together during group discussions.

[Roadmap](https://gist.github.com/aleksandr-dzhumurat/c443ddd92652d7fa38cbaf8f4a9d1955)

We want to leverage both personal user preferences and group-induced preferences to find the best solution in a chat environment.

It is proposed that a collective is able to impact decision-makers in a wide range of ways depending on the described context (i.e. familial, friendly, or professional). Therefore, it stands to reason that numerous methods of obtaining an apt outcome can be employed in each situation.

# Papers research

## Event sources

- [Top Google SERP API, Search Engine Proxies, and Scraping Tools](https://www.scraperapi.com/blog/top-google-serp-api-search-engine-proxies-and-scraping-tools/)
- [Perplexity](https://www.perplexity.ai/)
- [Bridgify](https://api.bridgify.io/attractions/category-tree/)

## Group recommendations

- [Preference Elicitation for Group Recommender Systems](https://drive.google.com/file/d/1xuca4DI2cACDoJD0IlovDghqYLRVfbrl/view?usp=drivesdk)
  - group recommender systems: *long term* and *short term* (session-based) utility vectors,
- [Preference Networks and Non-Linear Preferences in Group Recommendations](https://drive.google.com/file/d/1x3HlSkya11_UqaBpbo_wRN75GBGwzU9b/view?usp=drivesdk)
  - non-linear preference aggregation strategies
- [Single User Group Recommendations](https://drive.google.com/file/d/1rt68N7AXYbvaQwZJ7Rq2CdPJ39nOTPrm/view?usp=drivesdk)
  - support users to find restaurants suitable for the group they belong to
- [Group Dynamic and Group Recommender Systems for Decision Support](https://drive.google.com/file/d/1yG-nArxG308e9xZ60kjfSxpfQM5tVEuR/view?usp=drivesdk)
- [Conflict resolution in group decision making: insights from a simulation study](https://drive.google.com/file/d/1qY51tTfZsNSCuBWPlJlJylODrSOU4R15/view?usp=drivesdk)
- [An observational user study for group recommender systems in the tourism domain](https://drive.google.com/file/d/1xccaAOt684n-0l9mIJZWnBuREYSlXoM8/view?usp=drivesdk)
- [Combining Long-term and Discussion-generated Preferences in Group Recommendations](https://drive.google.com/file/d/1ofDsAbPwbGyR8gg7fNsRrt7Hg-RnyNWc/view?usp=drivesdk)

## Onboarding

- [Personality-Based Recommendation Methods for New Users](https://drive.google.com/file/d/1t4K_DWWihUXY-xrz1GgUfUMl1STT_apj/view?usp=drivesdk)
  - the new user cold-start problem
- [User Preference Elicitation, Rating Sparsity and Cold Start: Algorithms, Practical Challenges and Applications](https://drive.google.com/file/d/1oIin7xDeM31cLSVIj0F4I12EdwSjDqH2/view?usp=drivesdk)
- [Supporting Users in Finding Successful Matches in Reciprocal Recommender Systems](https://drive.google.com/file/d/14Ou2W77G4DzQckuAlIxS2WJGuztjRPqw/view?usp=drivesdk)

## Feed

- [Recommendations with Optimal Combination of Feature-Based and Item-Based Preferences](https://drive.google.com/file/d/1l3l8zQCArSVOvyYYowk_v8gl3619lSQB/view?usp=drivesdk)
  - leverage information about features of items (e.g., in the movie domain, movie genres, cast, etc.)
- [Learning to act: a Reinforcement Learning approach to recommend the best next activities](https://drive.google.com/file/d/1pqFpLHngAAFj-3gWkQxdbIbRv-zPzwMR/view?usp=drivesdk)
- [Inverse Reinforcement Learning and Point of Interest Recommendations](https://drive.google.com/file/d/10Oo2SjuhNN3xg2An5d54pjXcA3_f8cTw/view?usp=drivesdk)
- [Recommender Systems Effect on the Evolution of Users’ Choices Distribution](https://drive.google.com/file/d/1mTJX9r7wt4bS7dqVNUZQdpjFyv79pRE9/view?usp=drivesdk)
- [Popularity, novelty and relevance in point of interest recommendation: an experimental analysis](https://drive.google.com/file/d/1hijBdkZ2QvgB13RyqknF6GZ1H_vixy8S/view?usp=drivesdk)
- [Graph Learning based Recommender Systems: A Review](https://drive.google.com/file/d/19uUBzzMTGVr6gIyHMhrPlLOdcsGOcq7g/view?usp=drivesdk)
- [Prediction of Music Pairwise Preferences from Facial Expressions](https://drive.google.com/file/d/1XN8C0yYNZllDfPwZa52kuLVW23KhEkIa/view?usp=drivesdk)
- [Extracting Information or Resource? The Hotelling Rule Revisited under Asymmetric Information](https://drive.google.com/file/d/1hK7Gi9CQaxbAQAdreA_CvGtiW-0EaBXL/view?usp=drivesdk)

## Common

- [Information and Communication Technologies in Tourism 2021](https://drive.google.com/file/d/1hzncl_o-3VuljFqsxYjUWQmyhF9ONNA3/view?usp=drivesdk)
- [A Chat-Based Group Recommender](https://drive.google.com/file/d/1la4reTsmAoLbeVI30hbB_5FaqaXyLnBK/view?usp=drivesdk)
  - System for Tourism
- [Item Contents Good, User Tags Better: Empirical Evaluation of a Food Recommender System](https://drive.google.com/file/d/1ZXJ8ncffn8FxX7OyTaSd2uOUpXpFnxK2/view?usp=drivesdk)


# ML KPI’s & metrics

# Short list

Conversion (clicks) to “see all” (by category)

Clicks to the content page

Engagement

- Save
- shares
- Like
- Dislike
- purchases

Catalog coverage in recommendations

Intra-category content diversity (by genres/ other tags)

## Description

For the Aola app feed application with a machine learning recommender system, several product metrics can be used as KPIs to evaluate the system’s performance and effectiveness. Here are some relevant metrics:

1. Click-Through Rate (**CTR**): This metric measures the percentage of users who click on a recommended activity after it is shown to them. It helps evaluate how engaging and relevant the recommended activities are to the users.
2. Engagement Rate: This metric assesses how much time users spend interacting with the recommended activities, such as likes, comments, and shares. A higher engagement rate indicates that the recommended content is resonating well with the users.
3. Conversion Rate: If your activities feed application has specific conversion goals, such as driving users to a particular action (e.g., signing up for a premium subscription, sharing the app with friends), the conversion rate can be a vital metric to track the success of the recommender system in achieving these goals.
4. Virality: This metric measures how often activities recommended by the system are shared by users. The more a activitiy goes viral, the wider the reach of your app and the better the performance of the recommender system.
5. Retention Rate: This metric evaluates the percentage of users who continue to use the app over time after interacting with the recommended activitiy. A higher retention rate indicates that the recommended content keeps users engaged and interested in the application.
6. User Satisfaction Surveys: Conducting surveys to measure user satisfaction and feedback can provide valuable insights into how users perceive the recommended activities and the overall user experience.
7. Diversity of Content: Assessing the diversity of recommended activities can be crucial to avoid over-exposure of specific types of content and keep users interested in discovering new and fresh activities.
8. Serendipity: This metric evaluates how well the recommender system can introduce unexpected and delightful content to users, enhancing their overall experience.
9. Personalization Quality: If your recommender system provides personalized recommendations, you can measure the quality of personalization by evaluating how well the system understands individual user preferences and tailors the recommendations accordingly.
10. Negative Feedback Rate: Monitoring the rate at which users provide negative feedback (e.g., hiding or reporting activities) can help identify potential issues with the recommender system and content selection.
11. User “self-segmentation” to improve metrics in every segment. Directly ask user segment + make look-alike model to categorize users who not answer.

# Search queries

Use queries as a few-shot emaples


Restaurants:

- Trendy French restaurant in Chelsea opened at day time
- Hipster breakfast place in Midtown
- Modern Turkish cuisine
- High level (top rated) Italian restaurant for business meetings
- Wow food experience
- Romantic restaurant for the first date
- Best restaurant for kids birthday celebration
- Cozy breakfast restaurants
- Great coffee place around 5th avenue and 47th street
- Latest trendy restaurants in Downtown Manhattan
- Great bars to meet people
- Wow bar experiences
- Top roof restaurants and bars
- Afternoon 5 o’clock cozy tea rooms with best deserts

Movies:

- New thriller movie for tonight
- Romantic comedy starting from 7 till 8 pm around Union Square
- A new movie with Tom Cruise - movie theaters and showtimes
- Best movie theater with convenient seats in Tribeca area

Art:

- Best modern art galleries in Midtown
- Immersive contemporary art experience
- Sensual ballet and performance art
- Classical treasures of European art of XIX and XX centuries

Shows:

- Bubble shows, magic tricks and other fun shows
- Off Broadway most intriguing and prominent performances
- Day-time shows for kids
- Late night romantic and provocative shows
- Great wow shows
- Unique shows in Lower Manhattan
- Light and laser show on weekend

Kids

- how to entertain kids of 3-4 (6-7, 10) years old around Greenwich village in the morning
- Great discovery and learning experience for a boy 8 years old
- Fun things to do for family with kids on Monday in NYC
- Movies and shows for kids around 3 pm
- Entertainment centers for kids with trampolines and climbing walls
- Learning and fun places for kids development

Things to do and tours:

- Open air activities for family on Sunday morning
- Best walking routs in London for enjoying the city
- Scooter tours to discover hidden treasures of London
- Hop on - Hop off bus tours
- Food experiences tours
- Most unusual things you can do in New York to have fun
- Most unusual things you can do in New York to have fun on Saturday evening/night
- Crazy night life

Sports

- Tennis matches to watch in London within next 3 days
- Schedule of MMF events in New York

# Prompt engineering

# Test Queries (By city)

- Promt

```
    You are live in Barcelona
    
    Imagine 40 google queries useful for planning entrertainments: performances, art galleries, dance shows, dancing clubs, theatres, sport, kids activities, food, music concerts
    
    for family or single.
    
    Try to make as diverse as you can.
    
    For yong person and mature
    
    Every query should be between 5 and 10 words long
```

## City-specific queries

- Barcelona
    
    "Upcoming art exhibitions in Barcelona December 2023" "Best dance shows near me this weekend Barcelona" "Top-rated family-friendly theaters in Barcelona" "Live music concerts schedule Barcelona December 2023" "Contemporary art galleries open today in Barcelona" "Local salsa dancing clubs for beginners in Barcelona" "Kids-friendly activities in Barcelona holiday season 2023" "Mature audience comedy shows Barcelona this week" "Barcelona sports events calendar December 2023" "Authentic tapas restaurants with live music Barcelona" "Cultural performances for young adults in Barcelona" "Late-night jazz clubs with a cozy atmosphere Barcelona" "Outdoor activities for families in Barcelona parks" "Classic theater productions for all ages in Barcelona" "Barcelona beachfront dance parties schedule 2023" "Interactive art installations for kids in Barcelona" "Alternative music concerts for young crowd Barcelona" "Historical theaters showcasing Spanish plays Barcelona" "Contemporary dance performances for mature audience Barcelona" "Craft beer and live acoustic music venues Barcelona" "Family-friendly museums with interactive exhibits Barcelona" "Flamenco dance shows for tourists in Barcelona" "Barcelona indoor rock climbing facilities for all ages" "Ballet performances in historic venues Barcelona" "Live indie bands playing in Barcelona tonight" "Children's theater with puppet shows Barcelona December" "Rooftop bars with skyline views and live DJ Barcelona" "Science museums with hands-on activities for kids Barcelona" "Smooth jazz and wine bars in Barcelona city center" "Outdoor sports events suitable for families in Barcelona" "Dinner theaters offering immersive experiences in Barcelona" "Latin dance classes for singles in Barcelona" "Funky soul music concerts in intimate venues Barcelona" "Playgrounds and parks for toddlers in Barcelona" "Mature audience magic shows in historic venues Barcelona" "Electronic music festivals happening in Barcelona 2023" "Authentic Spanish cuisine cooking classes Barcelona" "Family-friendly cycling routes in and around Barcelona" "Shakespearean plays in English for expats in Barcelona" "Salsa and bachata dance festivals in Barcelona 2023"
    
- Berlin
    
    Best restaurants in Berlin with local and international cuisine. Cafes with Wi-Fi in Berlin for work or relaxation. Explore vegetarian restaurants in Berlin for delicious plant-based meals. Experience traditional German food in Berlin's authentic eateries. Discover current theater shows in Berlin for an entertaining evening. Enjoy live music events in Berlin for a vibrant nightlife. Cheer for your team at sports bars in Berlin with a lively atmosphere. Engage in outdoor sports activities at Berlin's scenic locations. Explore indoor sports facilities in Berlin for fitness and fun. Discover the best running routes in Berlin for a scenic jog. Family-friendly restaurants in Berlin with tasty options for all. Exciting kids' activities in Berlin for a day of family fun. Explore children's museums in Berlin for interactive learning experiences. Visit zoos in Berlin to witness diverse wildlife species. Relax in public parks in Berlin for leisure and picnics. Find inspiration in art galleries across Berlin's vibrant cultural scene. Explore historical landmarks in Berlin for a journey through time. Delight in captivating dance performances in Berlin's artistic venues. Experience film festivals in Berlin for a cinematic adventure. Discover diverse flavors at street food markets in Berlin. Join dance classes in Berlin to learn various dance styles. Attend contemporary dance shows in Berlin for artistic expression. Experience classic ballet performances in Berlin's cultural venues. Enjoy outdoor movie nights in Berlin's parks under the stars. Explore art house cinemas in Berlin for unique film experiences. Check the film screenings calendar in Berlin for upcoming movies. Immerse yourself in interactive theater experiences in Berlin. Attend movie premieres in Berlin for a glimpse of cinematic excellence. Witness experimental dance performances pushing artistic boundaries in Berlin. Visit classic movie theaters in Berlin for a nostalgic film experience. Discover hidden gem restaurants in Berlin for unique dining experiences. Find waterfront dining spots in Berlin for scenic views during meals. Explore restaurants in Berlin offering healthy and nutritious food options. Embark on epicurean culinary tours in Berlin for a gastronomic adventure. Experience kayaking adventures in Berlin's waterways for outdoor fun. Cycle through urban routes in Berlin for a scenic and active day. Visit artisanal food markets in Berlin for locally crafted culinary delights. Attend soccer matches in Berlin to cheer for your favorite team. Practice yoga in Berlin's parks for a serene and rejuvenating experience. Take a street art walking tour in Berlin to explore vibrant urban art. Attend food festivals in Berlin for a diverse culinary celebration. Engage in adrenaline-pumping adventure sports in Berlin's activity centers. Enjoy craft beer tasting in Berlin's breweries for unique beer flavors. Play badminton in Berlin's sports facilities for a friendly match. Explore photography exhibitions in Berlin for visual storytelling inspiration. Explore delightful ice cream shops in Berlin for sweet treats. Participate in running events in Berlin for a fitness challenge. Visit the Museum of Modern Art in Berlin for contemporary art collections. Embark on sailing adventures in Berlin's lakes for a nautical experience. Enjoy live jazz performances in Berlin for a musical evening
    
- Dubai
    
    "Contemporary art exhibitions Dubai December 2023" "Dubai family-friendly theaters schedule this weekend" "Arabic dance shows in Dubai cultural district" "Live music concerts schedule Dubai December 2023" "Explore local dance clubs for singles in Dubai" "Kids-friendly activities in Dubai theme parks 2023" "Mature audience comedy shows Dubai this week" "Dubai sports events calendar December 2023" "International cuisine restaurants with live music Dubai" "Cultural performances for young adults in Dubai" "Late-night jazz clubs with skyline views Dubai" "Outdoor activities for families in Dubai parks" "Classic theater productions for all ages in Dubai" "Dubai beachfront dance parties schedule 2023" "Interactive art installations for kids in Dubai" "Alternative music concerts for young crowd Dubai" "Historical theaters showcasing local plays Dubai" "Contemporary dance performances for mature audience Dubai" "Craft coffee shops with acoustic music Dubai" "Family-friendly museums with interactive exhibits Dubai" "Traditional dance shows for tourists in Dubai" "Dubai indoor rock climbing facilities for all ages" "Ballet performances in modern venues Dubai" "Live indie bands playing in Dubai tonight" "Children's theater with puppet shows Dubai December" "Rooftop bars with skyline views and live DJ Dubai" "Science museums with hands-on activities for kids Dubai" "Smooth jazz and wine bars in Dubai city center" "Outdoor sports events suitable for families in Dubai" "Dinner theaters offering immersive experiences in Dubai" "Latin dance classes for singles in Dubai" "Funky soul music concerts in intimate venues Dubai" "Playgrounds and parks for toddlers in Dubai" "Mature audience magic shows in historic venues Dubai" "Electronic music festivals happening in Dubai 2023" "Dubai culinary classes for international cuisines" "Family-friendly desert safaris and tours in Dubai" "Dubai cycling routes with scenic views for singles" "Shakespearean plays in English for expats in Dubai" "Salsa and bachata dance festivals in Dubai 2023"
    
- Singapore
    
    "Family-friendly theater shows in Singapore December 2023" "Contemporary art exhibitions happening now in Singapore galleries" "Top dance shows this weekend in Singapore venues" "Live music concerts schedule Singapore December 2023" "Explore cultural performances for all ages in Singapore" "Salsa dancing clubs for beginners in Singapore city" "Kids activities in Singapore during the holiday season" "Mature audience comedy shows Singapore this week" "Singapore sports events calendar December 2023" "Best local food markets with live music Singapore" "Cultural performances for young adults in Singapore" "Late-night jazz clubs with a view in Singapore" "Outdoor activities for families in Singapore parks" "Classic theater productions for all ages in Singapore" "Singapore beachfront dance parties schedule 2023" "Interactive art installations for kids in Singapore" "Alternative music concerts for young crowd Singapore" "Historical theaters showcasing Asian plays in Singapore" "Contemporary dance performances for mature audience Singapore" "Craft beer and live acoustic music venues Singapore" "Family-friendly museums with interactive exhibits Singapore" "Traditional dance shows for tourists in Singapore" "Indoor rock climbing facilities for all ages Singapore" "Ballet performances in historic venues Singapore" "Live indie bands playing in Singapore tonight" "Children's theater with puppet shows Singapore December" "Rooftop bars with skyline views and live DJ Singapore" "Science museums with hands-on activities for kids Singapore" "Smooth jazz and wine bars in Singapore city center" "Outdoor sports events suitable for families in Singapore" "Dinner theaters offering immersive experiences in Singapore" "Latin dance classes for singles in Singapore" "Funky soul music concerts in intimate venues Singapore" "Playgrounds and parks for toddlers in Singapore" "Mature audience magic shows in historic venues Singapore" "Electronic music festivals happening in Singapore 2023" "Authentic Asian cuisine cooking classes Singapore" "Family-friendly cycling routes in and around Singapore" "Shakespearean plays in English for expats in Singapore" "Salsa and bachata dance festivals in Singapore 2023"
