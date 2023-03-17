
#!/bin/bash
# 定义时间变量名和显示时间格式
datename=$(date +%Y%m%d-%H%M%S) 

echo "$datename"

20200516-150442



CREATE-IF:
```sql
CREATE UNIQUE INDEX IF NOT EXISTS some_index ON some_table(some_column, another_column);
```




Query on sqlite_master
```sql
SELECT count(*) FROM sqlite_master WHERE type="table" AND name = "查询的表名"
```



Select IF-ELSE:


```sql
SELECT *,
       CASE WHEN Password IS NOT NULL
       THEN 'Yes'
       ELSE 'No'
       END AS PasswordPresent
FROM myTable
```

```sql
select
'张三' as name,
case when
(
    -- 查询出来的数量或你的条件数据
    select 1
)>0 THEN '已完成' else '未完成' end as state
```