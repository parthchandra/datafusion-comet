-- Licensed to the Apache Software Foundation (ASF) under one
-- or more contributor license agreements.  See the NOTICE file
-- distributed with this work for additional information
-- regarding copyright ownership.  The ASF licenses this file
-- to you under the Apache License, Version 2.0 (the
-- "License"); you may not use this file except in compliance
-- with the License.  You may obtain a copy of the License at
--
-- http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing,
-- software distributed under the License is distributed on an
-- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
-- KIND, either express or implied.  See the License for the
-- specific language governing permissions and limitations
-- under the License.

select
    sum(if(d_day_name='Sunday', ss_sales_price, null)) sun_sales,
    sum(if(d_day_name='Monday', ss_sales_price, null)) mon_sales,
    sum(if(d_day_name='Tuesday', ss_sales_price, null)) tue_sales,
    sum(if(d_day_name='Wednesday', ss_sales_price, null)) wed_sales,
    sum(if(d_day_name='Thursday', ss_sales_price, null)) thu_sales,
    sum(if(d_day_name='Friday', ss_sales_price, null)) fri_sales,
    sum(if(d_day_name='Saturday', ss_sales_price, null)) sat_sales
from date_dim join store_sales on d_date_sk = ss_sold_date_sk
where d_year = 2000;