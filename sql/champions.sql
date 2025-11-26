WITH t1 AS (

SELECT  
    year,
    DriverId,
    sum(Points) AS total_points
FROM results    
GROUP BY 1,2
ORDER BY 1,3 DESC

),

t2 AS (

SELECT 
    *,
    ROW_NUMBER() OVER (PARTITION BY year ORDER BY total_points desc) AS rn_driver
FROM t1

)

SELECT 
    *
FROM t2
WHERE rn_driver = 1