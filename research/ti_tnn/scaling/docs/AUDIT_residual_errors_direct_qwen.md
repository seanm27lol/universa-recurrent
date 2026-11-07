# Residual errors of the direct Qwen3.5-4B proposal: an audit of the executable-but-wrong cases (CPU, saved records)

This audit reads only saved records. It made no model call, trained nothing, used no GPU and did not read the
reserved BIRD Mini-Dev 500.

**The cases.** In the semantic-proposal-repair study's frozen run (positions 192-255, seeds 200 and 201), the direct
proposal (D0) executed on 110 of 128 requests and was EX-correct on 74. That leaves **36 requests where the proposal
executed but disagreed with the benchmark answer**, covering 24 distinct questions. Twelve questions are wrong in
both seeds, four of them with identical SQL.

**Method.** For each case:
- the proposal and the reference SQL, both re-executed read-only on the pinned snapshot, with row counts and the first
  rows;
- a reading of the question and evidence against the schema.

Each case gets one or more labels, and "?" marks an uncertain label. A difference from the one reference query is not
treated as a semantic error by itself. Where the proposal answers the question as asked, or as well as the reference
does, the case is labelled **benchmark ambiguity**.

## Summary

| category | cases (of 36) | distinct questions (of 24) | of which uncertain |
|---|---|---|---|
| schema/value grounding | 17 | 11 | 0 |
| missing predicate | 3 | 2 | 2 |
| aggregation/grouping/ordering | 11 | 8 | 0 |
| joins | 8 | 7 | 0 |
| benchmark ambiguity | 16 | 11 | 6 |

Labels overlap. **6 cases (5 questions) are benchmark ambiguity alone**:
the proposal gives the same fact in another form, or answers the question better than the reference. For example:
- a yes/no answer against the reference's raw attribute;
- a count against a listed column;
- a ratio against a percentage;
- the capital's name against its id;
- the same person in another column layout.

No change to the model can fix these under EX. They bound what any repair could gain.

**What the categories show:**
- **Schema/value grounding** is the largest correctable group: 17 cases, 11 questions. It splits into:
  - **values the database stores differently:** 'Gilbert City' and 'true' against the stored 'Gilbert' and 'True';
    'GR' against 'GRC'; 'Czech Republic' against the stored 'Czech Rep.'; currency stored as text such as '$1,234';
  - **near-miss tables and columns:** award_category against award; Teams against TeamsSC; Extra_Runs against
    Ball_by_Ball; firstName against nameGiven; director.name, which holds the movie, against director.director;
    COUNT(*) where a votes column should be summed.

  Most of these are visible in the database itself, through a column's distinct values or its description, without
  the reference answer.
- **Aggregation/grouping/ordering** (11 cases, 8 questions) covers:
  - counting rows instead of distinct entities;
  - averaging identifiers instead of the evidence's formula;
  - integer division;
  - a reversed sort for "youngest";
  - the maximum per group against the global maximum.
- **Joins** (8 cases, 7 questions) covers:
  - fan-out through an unneeded table;
  - composite keys joined on one part (IMG_OBJ on OBJ_SAMPLE_ID alone);
  - a join on the losing team instead of the winner;
  - an inner join that silently drops rows.
- **Missing predicates** are rare and mostly entangled with table choice: 3 cases, 2 of them uncertain.

**Comparison with the earlier studies.** The thinking-mode Qwen3.5-4B with database tools answered 41 of 64 on
another population, against about 37 of 64 here without tools. Its tools included `lookup_values` and
`execute_select`. That fits the value-grounding cases being the ones tools address, but different populations cannot
establish it.

## The cases


### 1. bird23tf-00412 (seed 200, computer_student) — benchmark ambiguity

- **Question:** Is the teacher who teaches course no.9 a faculty member?
- **Evidence:** teacher refers to taughtBy.p_id; course no.9 refers to taughtBy.course_id = 9; faculty member refers to hasPosition ! = 0
- **Proposal:** `SELECT CASE WHEN p.hasPosition != '0' THEN 'Yes' ELSE 'No' END FROM taughtBy AS T1 INNER JOIN person AS p ON T1.p_id = p.p_id WHERE T1.course_id = 9`
- **Reference:** `SELECT T2.hasPosition FROM taughtBy AS T1 INNER JOIN person AS T2 ON T1.p_id = T2.p_id WHERE T1.course_id = 9`
- **Results:** the proposal gives 1 rows [['Yes']]; the reference gives 1 rows [['Faculty']]
- **Reading:** yes/no question; the proposal answers 'Yes' from hasPosition != '0', the reference returns the hasPosition value 'Faculty'. Same fact, different answer form

### 2. bird23tf-00439 (seed 200, talkingdata) — joins

- **Question:** How many events did the 88-years-old male users participate on 4th May,2016?
- **Evidence:** 88-years-old refers to age = 88; male refers to gender = 'M'; on 4th May, 2016 refers to timestamp LIKE '2016-05-04%';
- **Proposal:** `SELECT COUNT(*) FROM app_events AS T1 INNER JOIN events AS T2 ON T1.event_id = T2.event_id INNER JOIN gender_age AS T3 ON T2.device_id = T3.device_id WHERE T3.age = 88 AND T3.gender = 'M' AND T2.timestamp LIKE '2016-05-04%'`
- **Reference:** `SELECT COUNT(T2.event_id) FROM gender_age AS T1 INNER JOIN events AS T2 ON T2.device_id = T1.device_id WHERE T1.gender = 'M' AND SUBSTR(`timestamp`, 1, 10) = '2016-05-04' AND T1.age = 88`
- **Results:** the proposal gives 1 rows [['174']]; the reference gives 1 rows [['52']]
- **Reading:** an extra join to app_events multiplies each event by its apps: 174 against 52 events

### 3. bird23tf-00789 (seed 200, soccer_2016) — benchmark ambiguity

- **Question:** In the players, how many were out by hit wicket?
- **Evidence:** out by hit wicket refers to Out_Name = 'hit wicket'
- **Proposal:** `SELECT COUNT(P.Player_Name) FROM Wicket_Taken AS T1 INNER JOIN Out_Type AS T2 ON T1.Kind_Out = T2.Out_Id INNER JOIN Player AS P ON T1.Player_Out = P.Player_Id WHERE T2.Out_Name = 'hit wicket'`
- **Reference:** `SELECT Player_Out FROM Wicket_Taken AS T1 INNER JOIN Out_Type AS T2 ON T1.Kind_Out = T2.Out_Id WHERE Out_Name = 'hit wicket'`
- **Results:** the proposal gives 1 rows [['8']]; the reference gives 8 rows [['72'], ['113']]
- **Reading:** 'how many' answered with COUNT = 8; the reference lists the 8 Player_Out ids instead of a count

### 4. bird23tf-01497 (seed 200, public_review_platform) — aggregation/grouping/ordering, benchmark ambiguity

- **Question:** What is the attribute of the business with highest star rating?
- **Evidence:** highest star rating Max(stars); attribute of business refers to attribute_name
- **Proposal:** `SELECT attribute_name FROM Attributes INNER JOIN Business_Attributes ON Attributes.attribute_id = Business_Attributes.attribute_id WHERE business_id IN (SELECT business_id FROM Business ORDER BY stars DESC LIMIT 1)`
- **Reference:** `SELECT T3.attribute_name FROM Business AS T1 INNER JOIN Business_Attributes AS T2 ON T1.business_id = T2.business_id INNER JOIN Attributes AS T3 ON T2.attribute_id = T3.attribute_id ORDER BY T1.stars DESC LIMIT 1`
- **Results:** the proposal gives 7 rows [['Price Range'], ['By Appointment Only']]; the reference gives 1 rows [['Alcohol']]
- **Reading:** the proposal returns all 7 attributes of one top-starred business; the reference takes the first row of an ordered join (1 attribute). Many businesses tie at the top rating

### 5. bird23tf-01600 (seed 200, public_review_platform) — aggregation/grouping/ordering, benchmark ambiguity?

- **Question:** What is the category of the business with short review length and highest review stars within business ID from 7 to 14?
- **Evidence:** category of the business refers to category_name; short review length refers to review_length = 'Short'; the highest star rating refers to MAX(stars); business ID from 7 to 14 refers to business_id BETWEEN 7 AND 14
- **Proposal:** `SELECT T2.category_name FROM Reviews AS T1 JOIN Business_Categories AS T3 ON T1.business_id = T3.business_id JOIN Categories AS T2 ON T3.category_id = T2.category_id WHERE T1.review_length = 'Short' AND T1.business_id BETWEEN 7 AND 14 GROUP BY T1.business_id ORDER BY MAX(T1.review_stars) DESC LIMIT 1`
- **Reference:** `SELECT DISTINCT T3.category_name FROM Reviews AS T1 INNER JOIN Business_Categories AS T2 ON T1.business_id = T2.business_id INNER JOIN Categories AS T3 ON T2.category_id = T3.category_id WHERE T2.business_id >= 7 AND T2.business_id < 15 AND T1.review_length = 'Short' AND T1.review_stars = ( SELECT MAX(review_stars) FROM Reviews )`
- **Results:** the proposal gives 1 rows [['Shopping']]; the reference gives 6 rows [['Shopping'], ['Mobile Phones']]
- **Reading:** the proposal takes the highest rating per business within 7-14 and LIMIT 1 (1 category); the reference matches review_stars to the global MAX over all Reviews (6 categories)

### 6. bird23tf-01633 (seed 200, simpson_episodes) — schema/value grounding

- **Question:** What is the title of episode that won the Best International TV Series Award in 2017?
- **Evidence:** won refers to result = 'Winner'; in 2017 refers to year = 2017
- **Proposal:** `SELECT T1.title FROM Episode AS T1 INNER JOIN Award AS T2 ON T1.episode_id = T2.episode_id WHERE T2.result = 'Winner' AND T2.award_category = 'Best International TV Series' AND T2.year = 2017`
- **Reference:** `SELECT T2.title FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE SUBSTR(T1.year, 1, 4) = '2017' AND T1.award = 'Best International TV Series' AND T1.result = 'Winner';`
- **Results:** the proposal gives 0 rows []; the reference gives 1 rows [['MyPods and Boomsticks']]
- **Reading:** 'Best International TV Series' filtered on award_category; the value lives in award: 0 rows

### 7. bird23tf-01666 (seed 200, simpson_episodes) — aggregation/grouping/ordering, schema/value grounding

- **Question:** How many votes of 5-star did the episode "Lisa the Drama Queen" receive?
- **Evidence:** episode "Lisa the Drama Queen" refers to title = 'Lisa the Drama Queen';
- **Proposal:** `SELECT COUNT(*) AS vote_count FROM Vote WHERE episode_id = (SELECT episode_id FROM Episode WHERE title = 'Lisa the Drama Queen') AND stars = 5`
- **Reference:** `SELECT SUM(T2.votes) FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T1.title = 'Lisa the Drama Queen' AND T2.stars = 5;`
- **Results:** the proposal gives 1 rows [['1']]; the reference gives 1 rows [['134']]
- **Reading:** COUNT(*) of Vote rows (1) where the Vote table carries a votes column to SUM (134)

### 8. bird23tf-02957 (seed 200, image_and_language) — joins

- **Question:** How many white objects are there in image no.2347915?
- **Evidence:** white objects refers to ATT_CLASS = 'white'; image no.2347915 refers to IMG_ID = 2347915
- **Proposal:** `SELECT COUNT(*) FROM IMG_OBJ_ATT AS T1 INNER JOIN ATT_CLASSES AS T2 ON T1.ATT_CLASS_ID = T2.ATT_CLASS_ID INNER JOIN IMG_OBJ AS T3 ON T1.OBJ_SAMPLE_ID = T3.OBJ_SAMPLE_ID WHERE T2.ATT_CLASS = 'white' AND T1.IMG_ID = 2347915`
- **Reference:** `SELECT SUM(CASE WHEN T2.ATT_CLASS = 'white' THEN 1 ELSE 0 END) FROM IMG_OBJ_ATT AS T1 INNER JOIN ATT_CLASSES AS T2 ON T1.ATT_CLASS_ID = T2.ATT_CLASS_ID WHERE T1.IMG_ID = 2347915`
- **Results:** the proposal gives 1 rows [['84084']]; the reference gives 1 rows [['2']]
- **Reading:** IMG_OBJ joined on OBJ_SAMPLE_ID alone, which is unique only within an image: 84,084 against 2

### 9. bird23tf-02992 (seed 200, image_and_language) — aggregation/grouping/ordering

- **Question:** Calculate the average of object samples for the image.
- **Evidence:** DIVIDE(COUNT(OBJ_SAMPLE_ID), COUNT(IMG_ID));
- **Proposal:** `SELECT AVG(CAST(T1.OBJ_SAMPLE_ID AS REAL)) FROM IMG_OBJ AS T1`
- **Reference:** `SELECT CAST(COUNT(OBJ_SAMPLE_ID) AS REAL) / COUNT(DISTINCT IMG_ID) FROM IMG_OBJ`
- **Results:** the proposal gives 1 rows [['10.810700455896407']]; the reference gives 1 rows [['16.560561914672217']]
- **Reading:** AVG of the OBJ_SAMPLE_ID values, not the evidence's COUNT(OBJ_SAMPLE_ID) / COUNT(IMG_ID)

### 10. bird23tf-03000 (seed 200, image_and_language) — joins, benchmark ambiguity?

- **Question:** List all the attribute classes of the images that have a (5,5) coordinate.
- **Evidence:** attribute classes refer to ATT_CLASS; (5,5) coordinate refers to X and Y coordinates of the bounding box where X = 5 and Y = 5;
- **Proposal:** `SELECT DISTINCT T2.ATT_CLASS FROM IMG_OBJ_ATT AS T1 INNER JOIN ATT_CLASSES AS T2 ON T1.ATT_CLASS_ID = T2.ATT_CLASS_ID INNER JOIN IMG_OBJ AS T3 ON T1.OBJ_SAMPLE_ID = T3.OBJ_SAMPLE_ID WHERE T3.X = 5 AND T3.Y = 5`
- **Reference:** `SELECT T1.ATT_CLASS FROM ATT_CLASSES AS T1 INNER JOIN IMG_OBJ_ATT AS T2 ON T1.ATT_CLASS_ID = T2.ATT_CLASS_ID INNER JOIN IMG_OBJ AS T3 ON T2.IMG_ID = T3.IMG_ID WHERE T3.X = 5 AND T3.Y = 5`
- **Results:** the proposal gives 699 rows [['building s'], ['indoors']]; the reference gives 1640 rows [['silver'], ['silver']]
- **Reading:** object-level join on OBJ_SAMPLE_ID alone (an incomplete key) with DISTINCT; the reference joins at image level on IMG_ID without DISTINCT (1,640 rows with duplicates)

### 11. bird23tf-03089 (seed 200, hockey) — schema/value grounding, missing predicate?

- **Question:** Please list the first names of the coaches whose team played in 1922's Stanley Cup finals.
- **Evidence:** teams refer to tmID; year = 1922;
- **Proposal:** `SELECT M.firstName FROM Coaches C JOIN Teams T ON C.tmID = T.tmID AND C.year = T.year JOIN Master M ON C.coachID = M.coachID WHERE T.year = 1922`
- **Reference:** `SELECT T3.firstName FROM Coaches AS T1 INNER JOIN TeamsSC AS T2 ON T1.year = T2.year AND T1.tmID = T2.tmID INNER JOIN Master AS T3 ON T1.coachID = T3.coachID WHERE T2.year = 1922`
- **Results:** the proposal gives 12 rows [['Wes'], ['Leo']]; the reference gives 3 rows [['Pete'], ['Ken']]
- **Reading:** Teams instead of TeamsSC: every 1922 team, not the Stanley Cup finalists (12 against 3)

### 12. bird23tf-03140 (seed 200, world) — schema/value grounding

- **Question:** What are the official languages used in Greece?
- **Evidence:** official language refers to IsOfficial = 'T'; Greece is a name of country;
- **Proposal:** `SELECT Language FROM CountryLanguage WHERE CountryCode = 'GR' AND IsOfficial = 'T'`
- **Reference:** `SELECT T1.Language FROM CountryLanguage AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.Code WHERE T1.IsOfficial = 'T' AND T2.name = 'Greece'`
- **Results:** the proposal gives 0 rows []; the reference gives 1 rows [['Greek']]
- **Reading:** CountryCode = 'GR' guessed; Greece is 'GRC' (the reference joins Country by name): 0 rows

### 13. bird23tf-03553 (seed 200, food_inspection) — aggregation/grouping/ordering, joins

- **Question:** Among the businesses within the postal code 94117, what is total number of businesses with a high risk category?
- **Evidence:** businesses with a high risk category refer to business_id where risk_category = 'High Risk';
- **Proposal:** `SELECT COUNT(*) FROM businesses b JOIN violations v ON b.business_id = v.business_id WHERE b.postal_code = '94117' AND v.risk_category = 'High Risk'`
- **Reference:** `SELECT COUNT(DISTINCT T2.business_id) FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T2.postal_code = 94117 AND T1.risk_category = 'High Risk'`
- **Results:** the proposal gives 1 rows [['143']]; the reference gives 1 rows [['74']]
- **Reading:** COUNT(*) over businesses x violations counts violations (143); the question counts businesses (COUNT DISTINCT: 74)

### 14. bird23tf-03945 (seed 200, computer_student) — benchmark ambiguity

- **Question:** What is the ratio of professors and students?
- **Evidence:** professors refers to professor = 1; students refers to student = 1; ratio = divide(count(person.p_id) when professor = 1, count(person.p_id) when student = 1)
- **Proposal:** `SELECT CAST(SUM(CASE WHEN professor = 1 THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN student = 1 THEN 1 ELSE 0 END) AS ratio FROM person`
- **Reference:** `SELECT CAST(SUM(CASE WHEN professor = 1 THEN 1 ELSE 0 END) AS REAL) * 100 / SUM(CASE WHEN student = 1 THEN 1 ELSE 0 END) AS per FROM person`
- **Results:** the proposal gives 1 rows [['0.28703703703703703']]; the reference gives 1 rows [['28.703703703703702']]
- **Reading:** the evidence defines a ratio; the proposal returns it (0.287), the reference multiplies by 100 (28.7)

### 15. bird23tf-04247 (seed 200, professional_basketball) — aggregation/grouping/ordering

- **Question:** What is the first and last name of the player with the highest field goal made rate in 1973?
- **Evidence:** in 1973 refers to year = 1973; player with highest field goal made refers to Max(Divide(fgMade, fgAttempted))
- **Proposal:** `SELECT T1.firstName, T1.lastName FROM players AS T1 JOIN players_teams AS T2 ON T1.playerID = T2.playerID JOIN teams AS T3 ON T2.tmID = T3.tmID AND T2.year = T3.year WHERE T3.year = 1973 ORDER BY T2.fgMade / NULLIF(T2.fgAttempted, 0) DESC LIMIT 1`
- **Reference:** `SELECT T1.firstName, T1.lastName FROM players AS T1 INNER JOIN players_teams AS T2 ON T1.playerID = T2.playerID WHERE year = 1973 ORDER BY CAST(T2.fgMade AS REAL) / T2.fgAttempted DESC LIMIT 1`
- **Results:** the proposal gives 1 rows [['Kareem', 'Abdul-Jabbar']]; the reference gives 1 rows [['Walter', 'Brown']]
- **Reading:** fgMade / fgAttempted in integer arithmetic (SQLite truncates), so the ordering is wrong; the reference casts to REAL

### 16. bird23tf-04354 (seed 200, public_review_platform) — schema/value grounding

- **Question:** How many businesses are actively running in Gilbert City?
- **Evidence:** actively running refers to active = 'true';
- **Proposal:** `SELECT COUNT(*) FROM Business WHERE city = 'Gilbert City' AND active = 'true'`
- **Reference:** `SELECT COUNT(business_id) FROM Business WHERE city LIKE 'Gilbert' AND active LIKE 'True'`
- **Results:** the proposal gives 1 rows [['0']]; the reference gives 1 rows [['571']]
- **Reading:** the stored values are 'Gilbert' and 'True' (the reference uses LIKE); the proposal's 'Gilbert City' and 'true' match nothing: 0 against 571

### 17. bird23tf-04816 (seed 200, ice_hockey_draft) — schema/value grounding, benchmark ambiguity

- **Question:** Among all players drafted by the Toronto Maple Leafs, identify the percentage who are from Eastern Europe.
- **Evidence:** players refers to PlayerName; drafted by the Toronto Maple Leafs refers to overallby = 'Toronto Maple Leafs'; percentage = MULTIPLY(DIVIDE(SUM(nation = 'Eastern Europe'), COUNT(ELITEID) WHERE overallby = 'Toronto Maple Leafs'), 100); from Eastern Europe refers to nation in ('Belarus', 'Bulgaria', 'Czech Republic', 'Hungary', 'Moldova', 'Poland', 'Romania', 'Slovakia', 'Ukraine'); countries in a continent can be identified by referring to https://worldpopulationreview.com/country-rankings/list-of-countries-by-continent;
- **Proposal:** `SELECT CAST(SUM(IIF(PlayerInfo.nation IN ('Belarus', 'Bulgaria', 'Czech Republic', 'Hungary', 'Moldova', 'Poland', 'Romania', 'Slovakia', 'Ukraine'), 1, 0)) AS REAL) * 100 / COUNT(PlayerInfo.ELITEID) FROM PlayerInfo WHERE overallby = 'Toronto Maple Leafs'`
- **Reference:** `SELECT CAST(COUNT(CASE WHEN nation IN ('Belarus', 'Czech Rep.', 'Slovakia', 'Ukraine') THEN ELITEID ELSE NULL END) AS REAL) * 100 / COUNT(ELITEID) FROM PlayerInfo WHERE overallby = 'Toronto Maple Leafs'`
- **Results:** the proposal gives 1 rows [['9.090909090909092']]; the reference gives 1 rows [['24.242424242424242']]
- **Reading:** the proposal follows the evidence's country list, but the database stores 'Czech Rep.' and the reference uses a shorter list

### 18. bird23tf-04930 (seed 200, hockey) — joins, schema/value grounding, missing predicate

- **Question:** Name the goalies and season they played when Boston Bruins won number 1 in rank.
- **Evidence:** goalies refers to pos = 'G'; season refers to year

- **Proposal:** `SELECT M.lastName, M.firstName, T.year FROM Teams AS T JOIN Master AS M ON T.tmID = M.playerID WHERE T.name = 'Boston Bruins' AND T.rank = 1`
- **Reference:** `SELECT T1.firstName, T1.lastName, T3.year FROM Master AS T1 INNER JOIN Goalies AS T2 ON T1.playerID = T2.playerID INNER JOIN Teams AS T3 ON T2.year = T3.year AND T2.tmID = T3.tmID WHERE T1.deathYear IS NOT NULL AND T3.name = 'Boston Bruins' AND T3.rank = 1 AND T1.pos = 'G'`
- **Results:** the proposal gives 0 rows []; the reference gives 11 rows [['Frank', 'Brimsek', '1938'], ['Frank', 'Brimsek', '1939']]
- **Reading:** Teams joined to Master on tmID = playerID (meaningless), no Goalies table and no pos = 'G'. The reference also adds an unrequested deathYear IS NOT NULL

### 19. bird23tf-05527 (seed 200, disney) — joins, schema/value grounding

- **Question:** What is Disney's highest grossing action movie?
- **Evidence:** action movie refers to movie_title where genre = 'Action'; highest grossing movie refers to MAX(total_gross)
- **Proposal:** `SELECT T2.movie_title FROM movies_total_gross AS T1 INNER JOIN characters AS T2 ON T1.movie_title = T2.movie_title WHERE T1.genre = 'Action' ORDER BY CAST(REPLACE(T1.total_gross, '$', '') AS INTEGER) DESC LIMIT 1`
- **Reference:** `SELECT movie_title FROM movies_total_gross WHERE genre = 'Action' ORDER BY CAST(REPLACE(trim(total_gross, '$'), ',', '') AS REAL) DESC LIMIT 1`
- **Results:** the proposal gives 0 rows []; the reference gives 1 rows [['The Avengers']]
- **Reading:** an unneeded inner join to characters drops the movie; total_gross ('$1,234') cast without removing commas: 0 rows

### 20. bird23tf-05954 (seed 200, soccer_2016) — schema/value grounding, benchmark ambiguity?

- **Question:** How many overs were there in the first innings of match ID "335996"?
- **Evidence:** the first innings refers to Innings_No = 1; match ID "335996" refers to Match_Id = 335996
- **Proposal:** `SELECT COUNT(*) AS overs_count FROM Extra_Runs WHERE Match_Id = 335996 AND Innings_No = 1`
- **Reference:** `SELECT COUNT(Over_Id) FROM Ball_by_Ball WHERE Match_Id = 335996 AND Innings_No = 1`
- **Results:** the proposal gives 1 rows [['6']]; the reference gives 1 rows [['124']]
- **Reading:** Extra_Runs instead of Ball_by_Ball (6 against 124); the reference itself counts balls (COUNT(Over_Id)), not distinct overs

### 21. bird23tf-06494 (seed 200, hockey) — aggregation/grouping/ordering, schema/value grounding

- **Question:** Who is the youngest player who is still living. State the given name and date of birth.
- **Evidence:** still living refers to deathYear IS NULL; youngest refers to MAX(birthYear,birthMon,birthDay)
- **Proposal:** `SELECT firstName, birthYear || '/' || birthMon || '/' || birthDay AS birthDate FROM Master WHERE deathYear IS NULL ORDER BY CAST(birthYear AS INTEGER) ASC, CAST(birthMon AS INTEGER) ASC, CAST(birthDay AS INTEGER) ASC LIMIT 1`
- **Reference:** `SELECT nameGiven , nameGiven , birthYear, birthMon, birthDay FROM Master WHERE deathYear IS NULL ORDER BY birthYear DESC, birthMon DESC, birthday DESC LIMIT 1`
- **Results:** the proposal gives 1 rows [['Petr', 'None']]; the reference gives 1 rows [['None', 'None', '1993', '4']]
- **Reading:** 'youngest' sorted ascending (oldest first), and firstName instead of the given name (nameGiven)

### 22. bird23tf-01107 (seed 201, professional_basketball) — joins

- **Question:** In what year did the only team to beat the Houston in the final round of postseason series games earn its lowest ranking?
- **Evidence:** beat the Huston refers to tmIDLoser = 'HSM';  in final round of post season refers to round = 'DSF'
- **Proposal:** `SELECT T2.year FROM teams AS T2 INNER JOIN series_post AS T1 ON T2.tmID = T1.tmIDLoser WHERE T1.tmIDLoser = 'HSM' AND T1.round = 'DSF' AND (T1.tmIDWinner, T2.rank) IN ( SELECT tmIDWinner, MIN(rank) FROM series_post INNER JOIN teams ON teams.tmID = series_post.tmIDWinner WHERE round = 'DSF' GROUP BY tmIDWinner )`
- **Reference:** `SELECT T2.year FROM series_post AS T1 INNER JOIN teams AS T2 ON T1.tmIDWinner = T2.tmID WHERE T1.round = 'DSF' AND T1.tmIDLoser = 'HSM' ORDER BY T2.rank ASC LIMIT 1`
- **Results:** the proposal gives 0 rows []; the reference gives 1 rows [['1967']]
- **Reading:** teams joined on the loser (tmIDLoser) instead of the winner, with a row-value IN over a grouped MIN: 0 rows

### 23. bird23tf-01497 (seed 201, public_review_platform) — aggregation/grouping/ordering, benchmark ambiguity

- **Question:** What is the attribute of the business with highest star rating?
- **Evidence:** highest star rating Max(stars); attribute of business refers to attribute_name
- **Proposal:** `SELECT attribute_name FROM Business_Attributes INNER JOIN Attributes ON Business_Attributes.attribute_id = Attributes.attribute_id WHERE business_id IN ( SELECT business_id FROM Business ORDER BY stars DESC LIMIT 1 )`
- **Reference:** `SELECT T3.attribute_name FROM Business AS T1 INNER JOIN Business_Attributes AS T2 ON T1.business_id = T2.business_id INNER JOIN Attributes AS T3 ON T2.attribute_id = T3.attribute_id ORDER BY T1.stars DESC LIMIT 1`
- **Results:** the proposal gives 7 rows [['Price Range'], ['By Appointment Only']]; the reference gives 1 rows [['Alcohol']]
- **Reading:** as case 3 (seed 201)

### 24. bird23tf-01600 (seed 201, public_review_platform) — aggregation/grouping/ordering, benchmark ambiguity?

- **Question:** What is the category of the business with short review length and highest review stars within business ID from 7 to 14?
- **Evidence:** category of the business refers to category_name; short review length refers to review_length = 'Short'; the highest star rating refers to MAX(stars); business ID from 7 to 14 refers to business_id BETWEEN 7 AND 14
- **Proposal:** `SELECT C.category_name FROM Reviews AS R INNER JOIN Business_Categories AS BC ON R.business_id = BC.business_id INNER JOIN Categories AS C ON BC.category_id = C.category_id WHERE R.business_id BETWEEN 7 AND 14 AND R.review_length = 'Short' GROUP BY R.business_id ORDER BY MAX(R.review_stars) DESC LIMIT 1`
- **Reference:** `SELECT DISTINCT T3.category_name FROM Reviews AS T1 INNER JOIN Business_Categories AS T2 ON T1.business_id = T2.business_id INNER JOIN Categories AS T3 ON T2.category_id = T3.category_id WHERE T2.business_id >= 7 AND T2.business_id < 15 AND T1.review_length = 'Short' AND T1.review_stars = ( SELECT MAX(review_stars) FROM Reviews )`
- **Results:** the proposal gives 1 rows [['Shopping']]; the reference gives 6 rows [['Shopping'], ['Mobile Phones']]
- **Reading:** as case 4 (seed 201)

### 25. bird23tf-01633 (seed 201, simpson_episodes) — schema/value grounding

- **Question:** What is the title of episode that won the Best International TV Series Award in 2017?
- **Evidence:** won refers to result = 'Winner'; in 2017 refers to year = 2017
- **Proposal:** `SELECT T1.title FROM Episode AS T1 INNER JOIN Award AS T2 ON T1.episode_id = T2.episode_id WHERE T2.result = 'Winner' AND T2.year = 2017 AND T2.award_category = 'Best International TV Series'`
- **Reference:** `SELECT T2.title FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE SUBSTR(T1.year, 1, 4) = '2017' AND T1.award = 'Best International TV Series' AND T1.result = 'Winner';`
- **Results:** the proposal gives 0 rows []; the reference gives 1 rows [['MyPods and Boomsticks']]
- **Reading:** as case 5 (seed 201)

### 26. bird23tf-01840 (seed 201, disney) — schema/value grounding, aggregation/grouping/ordering, benchmark ambiguity?

- **Question:** Who directed the movie with the most voice actors?
- **Evidence:** who directed refers director;
- **Proposal:** `SELECT T1.name FROM director AS T1 INNER JOIN characters AS T2 ON T1.name = T2.movie_title INNER JOIN `voice-actors` AS T3 ON T2.movie_title = T3.movie GROUP BY T1.name ORDER BY COUNT(T3.character) DESC LIMIT 1`
- **Reference:** `SELECT T2.director, COUNT(DISTINCT T1.`voice-actor`) FROM `voice-actors` AS T1 INNER JOIN director AS T2 ON T1.movie = T2.name GROUP BY T2.director ORDER BY COUNT(DISTINCT T1.`voice-actor`) DESC LIMIT 1`
- **Results:** the proposal gives 1 rows [['Zootopia']]; the reference gives 1 rows [['Ron Clements', '92']]
- **Reading:** director.name holds the movie, director.director the person: the proposal returns a movie (Zootopia). It counts characters, not distinct voice actors; the reference also returns the count as a second column

### 27. bird23tf-02992 (seed 201, image_and_language) — aggregation/grouping/ordering

- **Question:** Calculate the average of object samples for the image.
- **Evidence:** DIVIDE(COUNT(OBJ_SAMPLE_ID), COUNT(IMG_ID));
- **Proposal:** `SELECT AVG(T2.OBJ_SAMPLE_ID) FROM IMG_OBJ AS T1 INNER JOIN IMG_OBJ_ATT AS T2 ON T1.OBJ_SAMPLE_ID = T2.OBJ_SAMPLE_ID WHERE T1.IMG_ID = (SELECT IMG_ID FROM IMG_OBJ_ATT GROUP BY IMG_ID HAVING COUNT(*) > 0 LIMIT 1)`
- **Reference:** `SELECT CAST(COUNT(OBJ_SAMPLE_ID) AS REAL) / COUNT(DISTINCT IMG_ID) FROM IMG_OBJ`
- **Results:** the proposal gives 1 rows [['9.312943752330506']]; the reference gives 1 rows [['16.560561914672217']]
- **Reading:** as case 8 (seed 201), restricted to one image

### 28. bird23tf-03000 (seed 201, image_and_language) — joins, benchmark ambiguity?

- **Question:** List all the attribute classes of the images that have a (5,5) coordinate.
- **Evidence:** attribute classes refer to ATT_CLASS; (5,5) coordinate refers to X and Y coordinates of the bounding box where X = 5 and Y = 5;
- **Proposal:** `SELECT DISTINCT T2.ATT_CLASS FROM IMG_OBJ_ATT AS T1 INNER JOIN ATT_CLASSES AS T2 ON T1.ATT_CLASS_ID = T2.ATT_CLASS_ID INNER JOIN IMG_OBJ AS T3 ON T1.OBJ_SAMPLE_ID = T3.OBJ_SAMPLE_ID WHERE T3.X = 5 AND T3.Y = 5`
- **Reference:** `SELECT T1.ATT_CLASS FROM ATT_CLASSES AS T1 INNER JOIN IMG_OBJ_ATT AS T2 ON T1.ATT_CLASS_ID = T2.ATT_CLASS_ID INNER JOIN IMG_OBJ AS T3 ON T2.IMG_ID = T3.IMG_ID WHERE T3.X = 5 AND T3.Y = 5`
- **Results:** the proposal gives 699 rows [['building s'], ['indoors']]; the reference gives 1640 rows [['silver'], ['silver']]
- **Reading:** as case 9 (seed 201)

### 29. bird23tf-03089 (seed 201, hockey) — schema/value grounding, missing predicate?

- **Question:** Please list the first names of the coaches whose team played in 1922's Stanley Cup finals.
- **Evidence:** teams refer to tmID; year = 1922;
- **Proposal:** `SELECT DISTINCT M.firstName FROM Coaches C JOIN Teams T ON C.tmID = T.tmID AND C.year = T.year JOIN Master M ON C.coachID = M.coachID WHERE T.year = 1922`
- **Reference:** `SELECT T3.firstName FROM Coaches AS T1 INNER JOIN TeamsSC AS T2 ON T1.year = T2.year AND T1.tmID = T2.tmID INNER JOIN Master AS T3 ON T1.coachID = T3.coachID WHERE T2.year = 1922`
- **Results:** the proposal gives 11 rows [['Wes'], ['Leo']]; the reference gives 3 rows [['Pete'], ['Ken']]
- **Reading:** as case 10 (seed 201)

### 30. bird23tf-03119 (seed 201, world) — benchmark ambiguity

- **Question:** Provide the name, capital city and its official language of the country with the highest life expectancy.
- **Evidence:** capital city refers to Capital; official language refers to IsOfficial = 'T'; highest life expectancy refers to MAX(LifeExpectancy);
- **Proposal:** `SELECT c.Name, ci.Name, cl.Language FROM Country AS c JOIN City AS ci ON c.Capital = ci.ID JOIN CountryLanguage AS cl ON c.Code = cl.CountryCode AND cl.IsOfficial = 'T' ORDER BY c.LifeExpectancy DESC LIMIT 1`
- **Reference:** `SELECT T1.Name, T1.Capital, T2.Language FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode INNER JOIN City AS T3 ON T1.Code = T3.CountryCode WHERE T2.IsOfficial = 'T' ORDER BY T1.LifeExpectancy DESC LIMIT 1`
- **Results:** the proposal gives 1 rows [['Andorra', 'Andorra la Vella', 'Catalan']]; the reference gives 1 rows [['Andorra', '55', 'Catalan']]
- **Reading:** the proposal returns the capital city's name (Andorra la Vella); the reference returns the Capital id (55). The proposal answers the question as asked

### 31. bird23tf-03140 (seed 201, world) — schema/value grounding

- **Question:** What are the official languages used in Greece?
- **Evidence:** official language refers to IsOfficial = 'T'; Greece is a name of country;
- **Proposal:** `SELECT Language FROM CountryLanguage WHERE CountryCode = 'GR' AND IsOfficial = 'T'`
- **Reference:** `SELECT T1.Language FROM CountryLanguage AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.Code WHERE T1.IsOfficial = 'T' AND T2.name = 'Greece'`
- **Results:** the proposal gives 0 rows []; the reference gives 1 rows [['Greek']]
- **Reading:** as case 11 (seed 201)

### 32. bird23tf-03945 (seed 201, computer_student) — benchmark ambiguity

- **Question:** What is the ratio of professors and students?
- **Evidence:** professors refers to professor = 1; students refers to student = 1; ratio = divide(count(person.p_id) when professor = 1, count(person.p_id) when student = 1)
- **Proposal:** `SELECT CAST(SUM(CASE WHEN professor = 1 THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN student = 1 THEN 1 ELSE 0 END) AS ratio FROM person`
- **Reference:** `SELECT CAST(SUM(CASE WHEN professor = 1 THEN 1 ELSE 0 END) AS REAL) * 100 / SUM(CASE WHEN student = 1 THEN 1 ELSE 0 END) AS per FROM person`
- **Results:** the proposal gives 1 rows [['0.28703703703703703']]; the reference gives 1 rows [['28.703703703703702']]
- **Reading:** as case 13 (seed 201)

### 33. bird23tf-04354 (seed 201, public_review_platform) — schema/value grounding

- **Question:** How many businesses are actively running in Gilbert City?
- **Evidence:** actively running refers to active = 'true';
- **Proposal:** `SELECT COUNT(*) FROM Business WHERE active = 'true' AND city = 'Gilbert City'`
- **Reference:** `SELECT COUNT(business_id) FROM Business WHERE city LIKE 'Gilbert' AND active LIKE 'True'`
- **Results:** the proposal gives 1 rows [['0']]; the reference gives 1 rows [['571']]
- **Reading:** as case 15 (seed 201)

### 34. bird23tf-04816 (seed 201, ice_hockey_draft) — schema/value grounding, benchmark ambiguity

- **Question:** Among all players drafted by the Toronto Maple Leafs, identify the percentage who are from Eastern Europe.
- **Evidence:** players refers to PlayerName; drafted by the Toronto Maple Leafs refers to overallby = 'Toronto Maple Leafs'; percentage = MULTIPLY(DIVIDE(SUM(nation = 'Eastern Europe'), COUNT(ELITEID) WHERE overallby = 'Toronto Maple Leafs'), 100); from Eastern Europe refers to nation in ('Belarus', 'Bulgaria', 'Czech Republic', 'Hungary', 'Moldova', 'Poland', 'Romania', 'Slovakia', 'Ukraine'); countries in a continent can be identified by referring to https://worldpopulationreview.com/country-rankings/list-of-countries-by-continent;
- **Proposal:** `SELECT CAST(SUM(IIF(nation IN ('Belarus', 'Bulgaria', 'Czech Republic', 'Hungary', 'Moldova', 'Poland', 'Romania', 'Slovakia', 'Ukraine'), 1, 0)) AS REAL) * 100 / COUNT(ELITEID) FROM PlayerInfo WHERE overallby = 'Toronto Maple Leafs'`
- **Reference:** `SELECT CAST(COUNT(CASE WHEN nation IN ('Belarus', 'Czech Rep.', 'Slovakia', 'Ukraine') THEN ELITEID ELSE NULL END) AS REAL) * 100 / COUNT(ELITEID) FROM PlayerInfo WHERE overallby = 'Toronto Maple Leafs'`
- **Results:** the proposal gives 1 rows [['9.090909090909092']]; the reference gives 1 rows [['24.242424242424242']]
- **Reading:** as case 16 (seed 201)

### 35. bird23tf-05527 (seed 201, disney) — schema/value grounding

- **Question:** What is Disney's highest grossing action movie?
- **Evidence:** action movie refers to movie_title where genre = 'Action'; highest grossing movie refers to MAX(total_gross)
- **Proposal:** `SELECT movie_title FROM movies_total_gross WHERE genre = 'Action' ORDER BY total_gross DESC LIMIT 1`
- **Reference:** `SELECT movie_title FROM movies_total_gross WHERE genre = 'Action' ORDER BY CAST(REPLACE(trim(total_gross, '$'), ',', '') AS REAL) DESC LIMIT 1`
- **Results:** the proposal gives 1 rows [['Crimson Tide']]; the reference gives 1 rows [['The Avengers']]
- **Reading:** total_gross is text ('$1,234,567'); ordering it as text picks the wrong movie (Crimson Tide against The Avengers)

### 36. bird23tf-06494 (seed 201, hockey) — benchmark ambiguity

- **Question:** Who is the youngest player who is still living. State the given name and date of birth.
- **Evidence:** still living refers to deathYear IS NULL; youngest refers to MAX(birthYear,birthMon,birthDay)
- **Proposal:** `SELECT nameGiven, birthYear || '/' || birthMon || '/' || birthDay AS birthDate FROM Master WHERE deathYear IS NULL ORDER BY birthYear DESC, birthMon DESC, birthDay DESC LIMIT 1`
- **Reference:** `SELECT nameGiven , nameGiven , birthYear, birthMon, birthDay FROM Master WHERE deathYear IS NULL ORDER BY birthYear DESC, birthMon DESC, birthday DESC LIMIT 1`
- **Results:** the proposal gives 1 rows [['None', '1993/4/18']]; the reference gives 1 rows [['None', 'None', '1993', '4']]
- **Reading:** the same person (born 1993/4/18, nameGiven NULL): the proposal returns 2 columns, the reference 5 (nameGiven twice, then year, month, day). Only the column layout differs

## Proposed next experiment (not launched)

The largest correctable group is grounding against the database's own contents: stored value formats and near-miss
columns. The typed repair of PR #71 told the model which literals did not occur in the question. It did not show the
model the database's actual values or column descriptions.

**The proposal: database-evidence revision (V1).**
- **The policy:** after the shared direct proposal, give the model, from the snapshot alone and never the reference:
  - the distinct values nearest each literal in the proposal: case-insensitive and prefix matches, at most 5 per
    literal, from the compared column and same-named columns;
  - the descriptions and sample values of the columns the proposal uses, and of near-miss columns in the same tables;
  - a check for text columns used numerically (currency stored as text, for example).
- **The comparison:** V1 against D1 (ordinary revision), with the same proposal, seed, token limits, deadline and
  selection rule as PR #71, and D0 reported.
- **The rule:** the registered +8-of-128 rule, applied consistently, on a fresh, pinned, never-used population.
  Before any model call, that population's ambiguity-prone questions should be estimated with this audit's method on
  the reference queries alone, so the reachable ceiling is known.
- **Why it is distinct:** a tool-free, deterministic value-and-schema lookup. It tests whether the residual errors
  come from missing database evidence rather than reasoning. It is not a larger model, a retrained controller or
  another typed-feedback variant.
- **A cheaper first step, still on the CPU:** for the 17 grounding-labelled cases here, check whether the proposed
  lookup would have surfaced the right value or column. This is a coverage test of the evidence, not of the model,
  and should precede any GPU budget request.

## Limits

One annotator, 36 cases from one population and one model; the labels are judgements with the evidence shown, not a
benchmark. Gold-reference errors are included in "benchmark ambiguity" only where the case above shows them.
