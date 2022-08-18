#Q3
UPDATE contributions_project
SET timestamp = STR_TO_DATE(timestamp, '%m/%d/%Y %H:%i:%s');

ALTER TABLE contributions_project
modify column timestamp datetime;

SELECT company_id,timestamp,user_id
FROM contributions_project
ORDER BY company_id, timestamp asc, user_id;

#Q4
UPDATE contributions_project
SET user_creation_ts = STR_TO_DATE(user_creation_ts, '%m/%d/%Y %H:%i:%s');

ALTER TABLE contributions_project
modify column user_creation_ts datetime;

SELECT COUNT(*) / COUNT(distinct user_id) AS AvgContributions
FROM contributions_project
WHERE user_creation_ts BETWEEN '2021-01-01' AND '2021-01-10';

#Questions
SELECT * FROM contributions_project
WHERE referral_source = 'email'
AND industry != '';

SELECT * FROM contributions_project
WHERE industry = 'Banking'
AND referral_source != 'google_organic'
AND landing_page != 'position_page';

SELECT * FROM contributions_project
WHERE referral_source = 'linkedin_organic'
AND landing_page != 'org_chart';