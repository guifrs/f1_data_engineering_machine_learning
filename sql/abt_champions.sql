WITH tb_fs_drivers AS (

SELECT
    *
FROM
    feature_store_drivers
WHERE 1=1
AND avgPositionCurrentTemp IS NOT NULL
AND dtYear < 2025
ORDER BY DriverId, dtRef) ,
    
tb_join AS (

SELECT
    t1.*,
CASE WHEN t2.DriverId IS NOT NULL THEN 1 ELSE 0 END AS flChamp
FROM tb_fs_drivers AS t1

LEFT JOIN champions AS t2
ON t1.DriverId = t2.DriverId
AND t1.dtYear = t2.year

)

SELECT
    *
FROM tb_join
ORDER BY dtRef DESC, DriverId