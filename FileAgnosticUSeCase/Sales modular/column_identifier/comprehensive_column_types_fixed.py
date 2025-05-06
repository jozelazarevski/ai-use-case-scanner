"""
Comprehensive Column Type Detector

This module provides an enhanced column type detection system that can identify
a wide range of specialized column types across various domains using advanced
pattern matching, statistical analysis, and machine learning techniques.
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import string
from collections import Counter
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('column_detector')


class ComprehensiveColumnTypeDetector:
    """
    Enhanced column type detector that can identify a wide range of specialized column types
    across various domains with advanced pattern matching and statistical analysis.
    """
    
    def __init__(self, confidence_threshold=0.7, min_sample_size=100, max_sample_size=1000):
        """
        Initialize the comprehensive column type detector
        
        Args:
            confidence_threshold (float): Minimum confidence score (0-1) required to assign a type
            min_sample_size (int): Minimum number of values to sample from each column
            max_sample_size (int): Maximum number of values to sample from each column
        """
        # Configuration parameters
        self.confidence_threshold = confidence_threshold
        self.min_sample_size = min_sample_size
        self.max_sample_size = max_sample_size
        
        # Initialize all pattern dictionaries
        self._initialize_patterns()
        
        # Combined patterns dictionary for easy lookup
        self.all_patterns = {}
        self._compile_all_patterns()
        
        # Initialize results storage
        self.column_type_results = {}
        self.column_stats = {}
        
    def _initialize_patterns(self):
        """Initialize patterns for all column types"""
        # Core type patterns (covered in base implementation)
        self.core_patterns = self._initialize_core_patterns()
        
        # Customer-related patterns
        self.customer_patterns = self._initialize_customer_patterns()
        
        # Time-related patterns
        self.time_patterns = self._initialize_time_patterns()
        
        # Geographic/location patterns
        self.location_patterns = self._initialize_location_patterns()
        
        # Marketing/campaign patterns
        self.marketing_patterns = self._initialize_marketing_patterns()
        
        # Operational/fulfillment patterns
        self.operational_patterns = self._initialize_operational_patterns()
        
        # Temporal patterns (expanded from basic date detection)
        self.temporal_patterns = self._initialize_temporal_patterns()
        
        # Financial patterns (expanded from basic sales detection)
        self.financial_patterns = self._initialize_financial_patterns()
        
        # Product patterns (expanded from basic product detection)
        self.product_patterns = self._initialize_product_patterns()
        
        # Personnel patterns
        self.personnel_patterns = self._initialize_personnel_patterns()
        
        # Digital patterns
        self.digital_patterns = self._initialize_digital_patterns()
        
        # Statistical patterns
        self.statistical_patterns = self._initialize_statistical_patterns()
        
        # Technical patterns
        self.technical_patterns = self._initialize_technical_patterns()
        
        # International patterns
        self.international_patterns = self._initialize_international_patterns()
        
    def _compile_all_patterns(self):
        """Compile all patterns into a single dictionary for easy lookup"""
        pattern_dicts = [
            ('core', self.core_patterns),
            ('customer', self.customer_patterns),
            ('time', self.time_patterns),
            ('location', self.location_patterns),
            ('marketing', self.marketing_patterns),
            ('operational', self.operational_patterns),
            ('temporal', self.temporal_patterns),
            ('financial', self.financial_patterns),
            ('product', self.product_patterns),
            ('personnel', self.personnel_patterns),
            ('digital', self.digital_patterns),
            ('statistical', self.statistical_patterns),
            ('technical', self.technical_patterns),
            ('international', self.international_patterns)
        ]
        
        for prefix, pattern_dict in pattern_dicts:
            for key, value in pattern_dict.items():
                full_key = f"{prefix}_{key}" if prefix != 'core' else key
                self.all_patterns[full_key] = value
        """
    Add an analyze_column method to the ComprehensiveColumnTypeDetector class
    
    This patch adds the missing analyze_column method that's referenced in complete_.py.
    The method serves as the main entry point for analyzing columns.
    """
    
    # Add this method to the ComprehensiveColumnTypeDetector class in comprehensive_column_types.py
    
    def analyze_column(self, series, **kwargs):
        """
        Analyze a column to identify its data type and characteristics
        
        Args:
            series (pandas.Series): The column data to analyze
            **kwargs: Additional arguments (unused but included for compatibility)
            
        Returns:
            dict: Analysis results including type, confidence, and statistics
        """
        # Get basic statistics and type info
        result = {
            'type': 'unknown',
            'confidence': 0.0,
            'subtypes': {},
            'statistics': {}
        }
        
        # Analyze the series based on its dtype
        dtype_analysis = self._analyze_dtype(series)
        
        # Perform different analyses based on determined basic type
        if dtype_analysis['base_type'] == 'numeric':
            return self._analyze_numeric_column(series, series, dtype_analysis)
        elif dtype_analysis['base_type'] == 'string':
            return self._analyze_string_column(series, series, dtype_analysis)
        elif dtype_analysis['base_type'] == 'datetime':
            return self._analyze_datetime_column(series, series, dtype_analysis)
        elif dtype_analysis['base_type'] == 'boolean':
            return self._analyze_boolean_column(series, series, dtype_analysis)
        else:
            # Fallback for unknown types - collect basic statistics
            clean_series = series.dropna()
            
            result['statistics'] = {
                'count': len(clean_series),
                'nulls': len(series) - len(clean_series),
                'null_percentage': (len(series) - len(clean_series)) / len(series) if len(series) > 0 else 0,
                'unique_count': clean_series.nunique(),
                'unique_percentage': clean_series.nunique() / len(clean_series) if len(clean_series) > 0 else 0
            }
            
            return result
    
    def _initialize_core_patterns(self):
        """Initialize core type patterns (date, product, sales)"""
        return {
            'date': {
                'name_patterns': [
                    r'\b(?:color|colour|hue|shade|tint)\b',
                    r'\b(?:product|item)[\s_-]*(?:color|colour)\b',
                    r'\b(?:color|colour)[\s_-]*(?:name|code|value|option)\b'
                ],
                'value_patterns': [
                    r'^(?:red|blue|green|yellow|black|white|orange|purple|pink|brown|gray|grey|silver|gold)',
                    r'\b(?:date|time|day|timestamp)\b',
                    r'\b(?:order|invoice|transaction|sales|purchase)[\s_-]*(?:date|time|day|dt)\b',
                    r'\b(?:created|modified|entered|posted|recorded)[\s_-]*(?:date|time|on|at)\b'
                ],
                'format_patterns': {
                    'iso_date': r'^\d{4}-\d{2}-\d{2}$',                        # YYYY-MM-DD
                    'us_date': r'^(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})$',       # MM/DD/YYYY or DD/MM/YYYY
                    'year_month_day': r'^(\d{4})(\d{2})(\d{2})$',              # YYYYMMDD
                    'timestamp': r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}',         # YYYY-MM-DD HH:MM
                }
            },
            'product': {
                'name_patterns': [
                    r'\b(?:product|item|sku|merchandise|good|article|material)\b',
                    r'\b(?:product|item|article)[\s_-]*(?:id|code|number|no|identifier|key)\b'
                ],
                'id_patterns': [
                    r'^[A-Z]{1,3}[-_]?\d{3,}$',            # ABC123, A-123, etc.
                    r'^\d{3,}[-_]?[A-Z]{1,3}$',            # 123ABC, 123-A, etc.
                    r'^[A-Z0-9]{2,3}[-_]?\d{4,}$',         # AB1234, XYZ9876, etc.
                    r'^(?:[A-Z0-9]{2,5}[-_]){1,2}[A-Z0-9]{2,5}$'  # AB-12-XY, 123-ABC-45, etc.
                ]
            },
            'sales': {
                'name_patterns': [
                    r'\b(?:sales|revenue|amount|value|total|sum|selling)\b',
                    r'\b(?:total|gross)[\s_-]*(?:sales|revenue|income|amount|value|price)\b',
                    r'\b(?:extended|line)[\s_-]*(?:amount|price|total|value)\b'
                ]
            },
            'quantity': {
                'name_patterns': [
                    r'\b(?:quantity|qty|count|units|volume|number|pieces)\b',
                    r'\b(?:number|count)[\s_-]*(?:of|items|units|pieces)\b',
                    r'\b(?:item|unit|piece)[\s_-]*count\b'
                ],
                'value_patterns': [
                    r'^\d+$',  # Integer values
                ]
            },
            'price': {
                'name_patterns': [
                    r'\b(?:price|cost|rate|fee|charge)\b',
                    r'\b(?:unit|per[\s_-]*item)[\s_-]*(?:price|cost|value|amount)\b',
                    r'\b(?:price|cost)[\s_-]*(?:per|each|unit|item)\b'
                ]
            },
            'boolean': {
                'name_patterns': [
                    r'\b(?:is|has|should|can|will|flag|indicator|status|state)\b',
                    r'\b(?:is[\s_-]*(?:active|enabled|valid|canceled|approved|completed))\b',
                    r'\b(?:has[\s_-]*(?:approved|verified|validated|confirmed))\b'
                ],
                'value_patterns': [
                    r'^(?:0|1)$',
                    r'^(?:true|false)$',
                    r'^(?:yes|no)$',
                    r'^(?:y|n)$',
                    r'^(?:t|f)$'
                ]
            },
            'id': {
                'name_patterns': [
                    r'\b(?:id|identifier|key|code|number|num|no|#)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:id|identifier|key|code|number|num|no))\b',  # customer_id, order_id, etc.
                    r'\b(?:primary|foreign|unique)[\s_-]*(?:key|id|identifier)\b'
                ],
                'value_patterns': [
                    r'^\d{4,}$',  # 4+ digit numbers
                    r'^[A-Za-z]{1,3}[-_]?\d{3,}$',  # AB123, A-123
                    r'^\d{3,}[-_]?[A-Za-z]{1,3}$',  # 123AB, 123-A
                    r'^[A-Fa-f0-9]{8}-?[A-Fa-f0-9]{4}-?[A-Fa-f0-9]{4}-?[A-Fa-f0-9]{4}-?[A-Fa-f0-9]{12}$'  # UUID
                ]
            },
            'name': {
                'name_patterns': [
                    r'\b(?:name|title|label|caption|heading)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:name|title|label))\b',  # product_name, category_name, etc.
                    r'\b(?:full|display|short|long)[\s_-]*(?:name|title|label)\b'
                ]
            },
            'description': {
                'name_patterns': [
                    r'\b(?:description|desc|details|info|information|overview|summary)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:description|desc|details))\b',  # product_desc, item_details, etc.
                    r'\b(?:short|long|brief|full|detailed)[\s_-]*(?:description|desc|details)\b'
                ]
            },
            'category': {
                'name_patterns': [
                    r'\b(?:category|cat|type|group|class|classification|segment|tier|level)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:category|type|group|class))\b',  # product_category, item_type, etc.
                    r'\b(?:main|primary|secondary|sub)[\s_-]*(?:category|cat|type|group|class)\b'
                ]
            },
            'status': {
                'name_patterns': [
                    r'\b(?:status|state|condition|stage|phase|position|progress)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:status|state))\b',  # order_status, account_state, etc.
                    r'\b(?:current|previous|next|final)[\s_-]*(?:status|state|stage|phase)\b'
                ],
                'value_patterns': [
                    r'^(?:active|inactive|pending|completed|canceled|on[\s_-]*hold|new|processing|confirmed|shipped)$',
                    r'^(?:A|I|P|C|X|H|N|S)$'
                ]
            },
            'url': {
                'name_patterns': [
                    r'\b(?:url|uri|link|hyperlink|web[\s_-]*address|site)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:url|uri|link))\b',  # product_url, image_link, etc.
                    r'\b(?:http|https|ftp|website|webpage)[\s_-]*(?:link|url|address)?\b'
                ],
                'value_patterns': [
                    r'^(?:https?|ftp)://[^\s/$.?#].[^\s]*$',
                    r'^www\.[^\s/$.?#].[^\s]*$'
                ]
            },
            'email': {
                'name_patterns': [
                    r'\b(?:email|e[\s_-]*mail|mail)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:email|mail))\b',  # customer_email, contact_mail, etc.
                    r'\b(?:electronic|digital)[\s_-]*(?:mail|mailbox|address)\b'
                ],
                'value_patterns': [
                    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                ]
            },
            'phone': {
                'name_patterns': [
                    r'\b(?:phone|telephone|mobile|cell|contact)[\s_-]*(?:number|no|#)?\b',
                    r'\b(?:[a-z]+[\s_-]*(?:phone|telephone|mobile|cell))\b',  # customer_phone, contact_mobile, etc.
                    r'\b(?:work|home|business|fax)[\s_-]*(?:phone|telephone|number)\b'
                ],
                'value_patterns': [
                    r'^\+?[\d\s-\(\).]{7,}$'
                ]
            }
        }
    
    def _initialize_customer_patterns(self):
        """Initialize customer-related patterns"""
        return {
            'customer_id': {
                'name_patterns': [
                    r'\b(?:customer|client|buyer|account|contact)[\s_-]*(?:id|number|code|no|identifier|key)\b',
                    r'\b(?:cust|clnt|acct)[\s_-]*(?:id|code|num|no)\b',
                    r'\bcust[\s_-]*(?:#|no|num|code)\b'
                ],
                'value_patterns': [
                    r'^[A-Z]{1,3}-\d{4,10}$',
                    r'^CUST\d{4,}$',
                    r'^\d{5,10}$'  # Simple numeric customer IDs
                ]
            },
            'customer_name': {
                'name_patterns': [
                    r'\b(?:customer|client|buyer|account)[\s_-]*(?:name|title|fullname)\b',
                    r'\b(?:first|last|given|family|middle|full)[\s_-]*name\b',
                    r'\b(?:fname|lname|fullname)\b'
                ],
                'value_patterns': [
                    r'^[A-Z][a-z]+\s+[A-Z][a-z]+$',  # First Last format
                    r'^[A-Z][a-z]+,\s+[A-Z][a-z]+$'  # Last, First format
                ]
            },
            'customer_segment': {
                'name_patterns': [
                    r'\b(?:customer|client|buyer|account)[\s_-]*(?:segment|tier|group|level|type|category|cohort)\b',
                    r'\b(?:loyalty|vip|premium|gold|silver|bronze)[\s_-]*(?:tier|level|status|group)\b',
                    r'\b(?:segment|segmentation|cluster)[\s_-]*(?:id|code|name|group|value)?\b'
                ],
                'value_patterns': [
                    r'^(?:premium|standard|basic|gold|silver|bronze|platinum|vip|regular)$',
                    r'^tier[\s_-]?\d$',
                    r'^[A-D]$'  # Simple letter-based segments
                ]
            },
            'customer_acquisition': {
                'name_patterns': [
                    r'\b(?:customer|client|lead|prospect)[\s_-]*(?:acquisition|source|origin|channel)\b',
                    r'\b(?:acquisition|conversion|signup|registration|join|onboarding)[\s_-]*(?:source|channel|medium|campaign)\b',
                    r'\b(?:lead|referral|traffic)[\s_-]*source\b'
                ],
                'value_patterns': [
                    r'^(?:web|email|social|referral|organic|paid|direct|offline|store|call|event)$',
                    r'^(?:facebook|google|twitter|linkedin|instagram|tiktok|youtube)$'
                ]
            },
            'customer_ltv': {
                'name_patterns': [
                    r'\b(?:customer|client|buyer|account)[\s_-]*(?:ltv|lifetime[\s_-]*value|clv|value)\b',
                    r'\b(?:predicted|forecasted|estimated|projected)[\s_-]*(?:ltv|lifetime[\s_-]*value|clv)\b',
                    r'\b(?:ltv|clv|cltv)[\s_-]*(?:value|amount|score)?\b'
                ]
            },
            'customer_contact': {
                'name_patterns': [
                    r'\b(?:email|e-?mail|mail)[\s_-]*(?:address)?\b',
                    r'\b(?:phone|telephone|mobile|cell)[\s_-]*(?:number|no)?\b',
                    r'\bcontact[\s_-]*(?:details|info)\b'
                ],
                'value_patterns': {
                    'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                    'phone': r'^\+?[\d\s-\(\).]{7,}$'
                }
            },
            'customer_address': {
                'name_patterns': [
                    r'\b(?:billing|shipping|mailing|delivery|home|work|office)[\s_-]*address\b',
                    r'\b(?:street|avenue|road|boulevard|lane|plaza|building|apt|suite)[\s_-]*(?:address|number|no)?\b',
                    r'\b(?:address|addr)[\s_-]*(?:line[\s_-]*[1-3]|1|2|3)\b'
                ]
            },
            'customer_status': {
                'name_patterns': [
                    r'\b(?:customer|client|account)[\s_-]*(?:status|state|condition|standing)\b',
                    r'\b(?:active|inactive|suspended|pending|canceled|terminated|churned)[\s_-]*(?:status|flag|indicator)?\b',
                    r'\b(?:status|state)[\s_-]*(?:code|value|indicator)?\b'
                ],
                'value_patterns': [
                    r'^(?:active|inactive|pending|suspended|closed|terminated|on[\s_-]*hold|new|verified)$',
                    r'^(?:0|1|Y|N|A|I|P)$'  # Common status codes
                ]
            },
            'customer_age': {
                'name_patterns': [
                    r'\b(?:customer|client|user|member|account)[\s_-]*(?:age|years|yrs)\b',
                    r'\b(?:age|years|yrs)[\s_-]*(?:old)?\b',
                    r'\b(?:birth|dob|birth[\s_-]*date)[\s_-]*(?:date|day|year)?\b'
                ],
                'value_patterns': [
                    r'^(?:1[89]|[2-9][0-9])$',  # Ages 18-99
                    r'^\d{1,2}/\d{1,2}/\d{4}$',  # MM/DD/YYYY
                    r'^\d{4}-\d{2}-\d{2}$'  # YYYY-MM-DD
                ]
            },
            'customer_gender': {
                'name_patterns': [
                    r'\b(?:gender|sex|male|female)\b',
                    r'\b(?:customer|client|user|member|account)[\s_-]*(?:gender|sex)\b'
                ],
                'value_patterns': [
                    r'^(?:M|F|Male|Female|man|woman|non[\s_-]*binary|other)$',
                    r'^(?:m|f|male|female)$'
                ]
            },
            'customer_purchase_frequency': {
                'name_patterns': [
                    r'\b(?:customer|client|buyer|account)[\s_-]*(?:purchase|order|transaction)[\s_-]*(?:frequency|rate|count|number)\b',
                    r'\b(?:purchase|order|buy|transaction)[\s_-]*(?:frequency|rate|cadence|pattern|cycle)\b',
                    r'\b(?:frequency|times|count)[\s_-]*(?:of|per)[\s_-]*(?:purchase|order|transaction)\b'
                ]
            },
            'customer_churn_risk': {
                'name_patterns': [
                    r'\b(?:customer|client|account|churn|attrition)[\s_-]*(?:risk|score|probability|likelihood|potential)\b',
                    r'\b(?:churn|attrition|loss|cancellation|defection)[\s_-]*(?:risk|score|rate|prediction|probability)\b',
                    r'\b(?:risk|likelihood|probability)[\s_-]*(?:of|to)[\s_-]*(?:churn|cancel|leave|attrition)\b'
                ],
                'value_patterns': [
                    r'^(?:high|medium|low|H|M|L|critical|warning|safe)$',
                    r'^(?:0|[0-9]\.[0-9]+|[0-9]{1,2}|100)%?$'  # 0-100% or 0.0-1.0
                ]
            }
        }
    
    def _initialize_time_patterns(self):
        """Initialize time-related patterns"""
        return {
            'time_of_day': {
                'name_patterns': [
                    r'\b(?:time|hour|minute|second)[\s_-]*(?:of[\s_-]*day)?\b',
                    r'\b(?:timestamp|datetime|datetimeoffset)\b',
                    r'\b(?:clock|wall)[\s_-]*time\b'
                ],
                'value_patterns': [
                    r'^\d{1,2}:\d{2}(:\d{2})?([aApP][mM])?$',  # 3:30PM, 15:45, 9:20:15
                    r'^\d{1,2}[aApP][mM]$'  # 3PM, 11am
                ]
            },
            'duration': {
                'name_patterns': [
                    r'\b(?:duration|elapsed|length|span|interval|period)[\s_-]*(?:time|minutes|seconds|hours)?\b',
                    r'\b(?:time[\s_-]*spent|time[\s_-]*elapsed|time[\s_-]*taken)\b',
                    r'\b(?:session|call|visit|stay|usage)[\s_-]*(?:length|duration|time)\b'
                ],
                'value_patterns': [
                    r'^\d+:?\d*:?\d*$',  # 30, 1:30, 2:45:30
                    r'^\d+[\s_-]*(?:ms|sec|min|hr|day|wk|mo|yr)$'  # 30sec, 45min, 2hr
                ]
            },
            'delivery_time': {
                'name_patterns': [
                    r'\b(?:delivery|shipping|arrival|fulfillment|transit)[\s_-]*(?:time|date|timeframe|window|period|schedule)\b',
                    r'\b(?:estimated|actual|scheduled|promised|target)[\s_-]*(?:delivery|arrival|ship)[\s_-]*(?:date|time)\b',
                    r'\b(?:eta|etd|ata|atd)[\s_-]*(?:date|time)?\b'  # Estimated/Actual Time of Arrival/Departure
                ]
            },
            'lead_time': {
                'name_patterns': [
                    r'\b(?:lead|lag|processing|handling|turnaround|response)[\s_-]*time\b',
                    r'\b(?:time[\s_-]*to[\s_-]*(?:process|fulfill|complete|respond|ship|deliver))\b',
                    r'\b(?:sla|service[\s_-]*level[\s_-]*agreement)[\s_-]*(?:time|hours|days)?\b'
                ]
            },
            'time_zone': {
                'name_patterns': [
                    r'\b(?:time[\s_-]*zone|tz|timezone|time[\s_-]*offset|utc[\s_-]*offset)\b',
                    r'\b(?:local|server|system|user)[\s_-]*(?:time[\s_-]*zone|tz)\b'
                ],
                'value_patterns': [
                    r'^(?:UTC|GMT|EST|CST|MST|PST|EDT|CDT|MDT|PDT)[\+\-]?\d*$',
                    r'^[\+\-]\d{1,2}(?::\d{2})?$'  # +8, -5:30
                ]
            },
            'frequency': {
                'name_patterns': [
                    r'\b(?:frequency|interval|periodicity|recurrence|cycle|cadence)\b',
                    r'\b(?:daily|weekly|monthly|quarterly|yearly|annual)[\s_-]*(?:frequency|occurrence|schedule)?\b',
                    r'\b(?:times[\s_-]*per[\s_-]*(?:day|week|month|year|quarter))\b'
                ],
                'value_patterns': [
                    r'^(?:daily|weekly|biweekly|monthly|quarterly|annually|hourly|minutely)$',
                    r'^(?:every|once)[\s_-]+(?:day|week|month|year|hour|minute|second)$',
                    r'^\d+[\s_-]*(?:ms|sec|min|hr|day|wk|mo|yr)$'
                ]
            },
            'datetime_components': {
                'name_patterns': [
                    r'\b(?:year|yr|yyyy|fiscal[\s_-]*year|calendar[\s_-]*year)\b',
                    r'\b(?:month|mon|mm|quarter|qtr|season)\b',
                    r'\b(?:day|dy|dd|weekday|dow|dom)\b',
                    r'\b(?:hour|hr|minute|min|second|sec)\b'
                ],
                'value_patterns': {
                    'year': r'^(?:19|20)\d{2}$',  # 1900-2099
                    'month': r'^(?:0?[1-9]|1[0-2])$|^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$',
                    'day': r'^(?:0?[1-9]|[12][0-9]|3[01])$',
                    'hour': r'^(?:[01]?[0-9]|2[0-3])$',
                    'minute': r'^(?:[0-5]?[0-9])$',
                }
            },
            'week': {
                'name_patterns': [
                    r'\b(?:week|wk|week[\s_-]*number|week[\s_-]*no|weeknum)\b',
                    r'\b(?:iso|calendar|fiscal)[\s_-]*week\b',
                    r'\b(?:woy|wow|ww)[\s_-]*(?:number|no)?\b'  # Week of year, Week over week, Work week
                ],
                'value_patterns': [
                    r'^(?:[0-9]|[1-4][0-9]|5[0-3])$',  # 0-53
                    r'^W(?:[0-9]|[1-4][0-9]|5[0-3])$'  # W1-W53
                ]
            }
        }
    
    def _initialize_location_patterns(self):
        """Initialize geographic/location patterns"""
        return {
            'address': {
                'name_patterns': [
                    r'\b(?:address|addr|location|street|ave|blvd|road|rd)[\s_-]*(?:line)?[\s_-]*(?:1|2|3|one|two|three)?\b',
                    r'\b(?:billing|shipping|mailing|delivery|home|work|business)[\s_-]*address\b',
                    r'\b(?:street|building|location)[\s_-]*(?:address|number|name)\b'
                ]
            },
            'city': {
                'name_patterns': [
                    r'\b(?:city|town|municipality|borough|village|suburb|settlement|urban[\s_-]*area)\b',
                    r'\b(?:billing|shipping|mailing|delivery|home|work|business)[\s_-]*city\b',
                    r'\b(?:city|town)[\s_-]*(?:name|code)?\b'
                ]
            },
            'state_province': {
                'name_patterns': [
                    r'\b(?:state|province|county|region|district|territory|prefecture)\b',
                    r'\b(?:billing|shipping|mailing|delivery|home|work|business)[\s_-]*(?:state|province)\b',
                    r'\b(?:state|province)[\s_-]*(?:name|code|abbr|abbreviation)?\b'
                ],
                'value_patterns': [
                    r'^[A-Z]{2}$',  # US state codes: NY, CA, etc.
                    r'^(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)$'  # Specific US states
                ]
            },
            'postal_code': {
                'name_patterns': [
                    r'\b(?:zip|postal|post)[\s_-]*(?:code)?\b',
                    r'\b(?:billing|shipping|mailing|delivery|home|work|business)[\s_-]*(?:zip|postal|post)[\s_-]*(?:code)?\b'
                ],
                'value_patterns': [
                    r'^\d{5}(?:-\d{4})?$',  # US ZIP: 12345 or 12345-6789
                    r'^[A-Z]\d[A-Z][\s-]?\d[A-Z]\d$'  # Canadian postal code: A1A 1A1
                ]
            },
            'country': {
                'name_patterns': [
                    r'\b(?:country|nation|land|territory|commonwealth|republic|kingdom)\b',
                    r'\b(?:billing|shipping|mailing|delivery|home|work|business)[\s_-]*country\b',
                    r'\b(?:country|nation)[\s_-]*(?:name|code|iso)?\b'
                ],
                'value_patterns': [
                    r'^(?:US|USA|CA|CAN|UK|GB|AU|DE|FR|JP|CN|IN|BR)$',  # Common ISO codes
                    r'^[A-Z]{2}$',  # ISO 2-letter codes
                    r'^[A-Z]{3}$'   # ISO 3-letter codes
                ]
            },
            'region': {
                'name_patterns': [
                    r'\b(?:region|area|zone|district|sector|territory|jurisdiction)\b',
                    r'\b(?:sales|service|delivery|market)[\s_-]*(?:region|area|zone|territory)\b',
                    r'\b(?:region|zone)[\s_-]*(?:name|code|id)?\b'
                ]
            },
            'geo_coordinates': {
                'name_patterns': [
                    r'\b(?:latitude|lat|longitude|long|lon|lng|coords|coordinates|geo|gps|position)\b',
                    r'\b(?:lat|latitude)[\s_-]*(?:value|coordinate|position|degrees)?\b',
                    r'\b(?:lon|long|longitude)[\s_-]*(?:value|coordinate|position|degrees)?\b'
                ],
                'value_patterns': {
                    'latitude': r'^-?(?:90|[1-8]?[0-9](?:\.\d+)?)$',
                    'longitude': r'^-?(?:180|1[0-7][0-9]|[1-9]?[0-9](?:\.\d+)?)$'
                }
            },
            'store_location': {
                'name_patterns': [
                    r'\b(?:store|shop|outlet|branch|location|site|dealer|franchise)[\s_-]*(?:id|code|number|location|address)?\b',
                    r'\b(?:retail|warehouse|distribution|pickup|collection)[\s_-]*(?:location|site|center|facility)\b'
                ],
                'value_patterns': [
                    r'^[A-Z]{1,2}[-_]?\d{2,4}$',  # Store codes like NY-123, TX12
                    r'^ST\d{3,}$',  # ST123, ST4567
                    r'^\d{3,5}$'    # Simple numeric store IDs
                ]
            }
        }
    
    def _initialize_marketing_patterns(self):
        """Initialize marketing and campaign patterns"""
        return {
            'campaign': {
                'name_patterns': [
                    r'\b(?:campaign|promotion|promo|initiative|drive|marketing)[\s_-]*(?:id|code|name|number|no|identifier)?\b',
                    r'\b(?:advertisement|ad|advert)[\s_-]*(?:campaign|id|code|name|number)?\b',
                    r'\b(?:camp|cmpgn)[\s_-]*(?:id|code|num|no)?\b'
                ]
            },
            'channel': {
                'name_patterns': [
                    r'\b(?:channel|medium|platform|source|touchpoint|outlet)\b',
                    r'\b(?:marketing|sales|distribution|communication|acquisition)[\s_-]*(?:channel|source|medium)\b',
                    r'\b(?:utm|tracking)[\s_-]*(?:source|medium|channel)\b'
                ],
                'value_patterns': [
                    r'^(?:web|email|social|print|tv|radio|direct|mail|store|online|offline|mobile|app|referral)$',
                    r'^(?:facebook|twitter|instagram|linkedin|youtube|google|pinterest|tiktok)$'
                ]
            },
            'promotion': {
                'name_patterns': [
                    r'\b(?:promotion|promo|offer|deal|discount|special)[\s_-]*(?:id|code|name|type)?\b',
                    r'\b(?:coupon|voucher|rebate)[\s_-]*(?:code|id|number|amount|value|percent)?\b',
                    r'\b(?:promo|discount)[\s_-]*(?:rate|amount|value|percentage|level|tier)?\b'
                ]
            },
            'segment': {
                'name_patterns': [
                    r'\b(?:segment|segmentation|audience|cohort|group|cluster|category)\b',
                    r'\b(?:market|customer|user|buyer|client|visitor)[\s_-]*(?:segment|group|cluster|category|cohort)\b',
                    r'\b(?:demographic|psychographic|behavioral)[\s_-]*(?:segment|group|cluster|category)?\b'
                ]
            },
            'conversion': {
                'name_patterns': [
                    r'\b(?:conversion|click|action|engagement|interaction|event)[\s_-]*(?:rate|percentage|count|number|value)?\b',
                    r'\b(?:bounce|exit|abandonment)[\s_-]*rate\b',
                    r'\b(?:ctr|cvr|cpa|cpc|cpm)[\s_-]*(?:value|rate|amount)?\b'  # Click-through rate, Conversion rate, etc.
                ]
            },
            'attribution': {
                'name_patterns': [
                    r'\b(?:attribution|contribution|credit|source|origin|influence)\b',
                    r'\b(?:first|last|multi|linear|time|position|touch)[\s_-]*(?:touch|click|interaction|attribution|credit)\b',
                    r'\b(?:attribution|contribution)[\s_-]*(?:model|method|approach|algorithm|rule|logic)\b'
                ],
                'value_patterns': [
                    r'^(?:first|last|linear|time|position|multi|data)[\s_-]*(?:touch|click|interaction|attribution)$',
                    r'^(?:direct|organic|referral|social|email|paid|affiliate|partner)$'
                ]
            },
            'marketing_cost': {
                'name_patterns': [
                    r'\b(?:marketing|advertising|promotion|campaign)[\s_-]*(?:cost|expense|spend|budget|investment)\b',
                    r'\b(?:ad|campaign|promotion)[\s_-]*(?:spend|cost|expense|budget)\b',
                    r'\b(?:cac|cpa|cpc|cpm|cpp)[\s_-]*(?:cost|value|amount|rate)?\b'  # Cost per acquisition, Cost per click, etc.
                ]
            },
            'marketing_roi': {
                'name_patterns': [
                    r'\b(?:marketing|advertising|promotion|campaign)[\s_-]*(?:roi|return|roas|performance|efficiency)\b',
                    r'\b(?:roi|roas|romi|return)[\s_-]*(?:on|of)[\s_-]*(?:marketing|advertising|ad|campaign|promotion)[\s_-]*(?:spend|investment|expense)?\b',
                    r'\b(?:campaign|ad|promotion)[\s_-]*(?:roi|return|performance|results|success|effectiveness)\b'
                ],
                'value_patterns': [
                    r'^-?\d+(?:\.\d+)?%?$',  # Numeric ROI (possibly with percentage sign)
                    r'^-?\d+(?:\.\d+)?[xX]$'  # ROI in format like "3.5x"
                ]
            }
        }
    
    def _initialize_operational_patterns(self):
        """Initialize operational/fulfillment patterns"""
        return {
            'order_status': {
                'name_patterns': [
                    r'\b(?:order|shipment|delivery|fulfillment|transaction)[\s_-]*(?:status|state|condition|stage|phase)\b',
                    r'\b(?:current|latest|updated|tracking)[\s_-]*(?:status|state|condition)\b',
                    r'\b(?:status|state)[\s_-]*(?:code|value|indicator|flag)?\b'
                ],
                'value_patterns': [
                    r'^(?:new|pending|processing|shipped|delivered|completed|cancelled|returned|on[-\s_]*hold|back[-\s_]*ordered)$',
                    r'^(?:N|P|S|D|C|X|R|H|B)$'  # Status codes
                ]
            },
            'shipping_method': {
                'name_patterns': [
                    r'\b(?:shipping|delivery|transport|carrier|shipment|fulfillment)[\s_-]*(?:method|type|mode|service|option|provider|carrier|company)?\b',
                    r'\b(?:express|standard|overnight|priority|ground|air|freight)[\s_-]*(?:shipping|delivery|service)?\b',
                    r'\b(?:ship|delivery)[\s_-]*(?:via|by|through|method|type|mode)\b'
                ],
                'value_patterns': [
                    r'^(?:standard|express|overnight|priority|ground|air|2[-\s_]*day|next[-\s_]*day)$',
                    r'^(?:fedex|ups|usps|dhl|amazon|royal[-\s_]*mail|canada[-\s_]*post)$'
                ]
            },
            'tracking_id': {
                'name_patterns': [
                    r'\b(?:tracking|shipment|package|parcel|delivery)[\s_-]*(?:id|number|code|identifier|reference|no)\b',
                    r'\b(?:track|trace)[\s_-]*(?:id|no|number|code)\b',
                    r'\b(?:waybill|airway[-\s_]*bill|bill[-\s_]*of[-\s_]*lading)[\s_-]*(?:number|no|id|code)?\b'
                ],
                'value_patterns': [
                    r'^[0-9]{8,15}$',  # Basic numeric tracking
                    r'^[A-Z]{2}[0-9]{9}[A-Z]{2}$',  # USPS format
                    r'^1Z[A-Z0-9]{16}$'  # UPS format
                ]
            },
            'fulfillment_center': {
                'name_patterns': [
                    r'\b(?:fulfillment|distribution|warehouse|storage|inventory|logistics)[\s_-]*(?:center|facility|location|building|site|hub|depot)\b',
                    r'\b(?:fc|dc|wh)[\s_-]*(?:id|code|number|name|location)?\b',
                    r'\b(?:shipping|fulfillment|shipping)[\s_-]*(?:from|origin|source|location)\b'
                ]
            },
            'inventory_status': {
                'name_patterns': [
                    r'\b(?:inventory|stock|supply|availability|quantity)[\s_-]*(?:status|level|state|condition|position)\b',
                    r'\b(?:in[-\s_]*stock|out[-\s_]*of[-\s_]*stock|available|unavailable|backorder)[\s_-]*(?:status|flag|indicator)?\b',
                    r'\b(?:stock|inventory)[\s_-]*(?:on[-\s_]*hand|available|reserved|allocated|committed)\b'
                ],
                'value_patterns': [
                    r'^(?:in[-\s_]*stock|out[-\s_]*of[-\s_]*stock|low[-\s_]*stock|available|unavailable|backorder)$',
                    r'^(?:1|0|Y|N|A|U|L|B)$'  # Status codes (1/0, Yes/No, etc.)
                ]
            },
            'batch_number': {
                'name_patterns': [
                    r'\b(?:batch|lot|production|manufacturing)[\s_-]*(?:number|no|id|code|identifier)\b',
                    r'\b(?:batch|lot)[\s_-]*(?:qty|quantity|volume|amount|count)?\b',
                    r'\b(?:production|process|manufacturing)[\s_-]*(?:run|sequence|series)[\s_-]*(?:id|no|number)?\b'
                ],
                'value_patterns': [
                    r'^[A-Z]{1,3}[0-9]{4,8}$',  # BA12345
                    r'^[0-9]{4,8}[A-Z]{1,3}$',  # 12345BA
                    r'^(?:B|L|P)-[0-9]{4,8}$'  # B-12345
                ]
            },
            'order_id': {
                'name_patterns': [
                    r'\b(?:order|purchase|transaction|invoice|sales|checkout)[\s_-]*(?:id|number|no|code|identifier|reference)\b',
                    r'\b(?:ord|po|pur|inv)[\s_-]*(?:id|no|number|code|#)\b',
                    r'\b(?:confirmation|receipt)[\s_-]*(?:number|no|id|code)\b'
                ],
                'value_patterns': [
                    r'^ORD-?\d{5,10}$',  # ORD12345, ORD-12345
                    r'^PO-?\d{5,10}$',   # PO12345, PO-12345
                    r'^\d{5,10}$'        # Simple numeric order IDs
                ]
            },
            'order_line': {
                'name_patterns': [
                    r'\b(?:order|line|item)[\s_-]*(?:line|item|position|number|sequence|sequence[\s_-]*number)\b',
                    r'\b(?:line|item)[\s_-]*(?:in|on|of)[\s_-]*(?:order|invoice|receipt)\b',
                    r'\b(?:item|position|line)[\s_-]*(?:#|no|number)\b'
                ],
                'value_patterns': [
                    r'^\d{1,3}$',  # Simple line numbers (1, 2, 3, etc.)
                    r'^\d{1,3}\.\d{1,3}$'  # Hierarchical line numbers (1.1, 1.2, etc.)
                ]
            }
        }
    
    def _initialize_temporal_patterns(self):
        """Initialize temporal patterns (expanded date/time detection)"""
        return {
            'year': {
                'name_patterns': [
                    r'\b(?:year|yr|yyyy|fiscal[\s_-]*year|fy|annual|calendar[\s_-]*year)\b',
                    r'\b(?:year|yr)[\s_-]*(?:of|in)[\s_-]*(?:order|purchase|transaction|sale|report)\b'
                ],
                'value_patterns': [
                    r'^(?:19|20)\d{2}$',  # 1900-2099
                    r'^\d{2}$',  # 2-digit years (22, 23, etc.)
                    r'^FY\d{2}(?:\d{2})?$'  # Fiscal year formats (FY22, FY2022)
                ]
            },
            'month': {
                'name_patterns': [
                    r'\b(?:month|mon|mm|mo|mnth)\b',
                    r'\b(?:month|mon)[\s_-]*(?:of|in)[\s_-]*(?:order|purchase|transaction|sale|year|report)\b',
                    r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(?:uary|ruary|ch|il|ust|tember|ober|ember)?\b'
                ],
                'value_patterns': [
                    r'^(?:0?[1-9]|1[0-2])$',  # 1-12 (or 01-12)
                    r'^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$',  # Three-letter abbreviations
                    r'^(?:January|February|March|April|May|June|July|August|September|October|November|December)$'  # Full names
                ]
            },
            'quarter': {
                'name_patterns': [
                    r'\b(?:quarter|qtr|q[1-4]|fiscal[\s_-]*quarter)\b',
                    r'\b(?:quarter|qtr)[\s_-]*(?:[1-4]|one|two|three|four)\b',
                    r'\b(?:q|quarter)[\s_-]*(?:of|in)[\s_-]*(?:year|fiscal|fy)\b'
                ],
                'value_patterns': [
                    r'^[Qq][1-4]$',  # Q1, Q2, Q3, Q4 (or q1, q2, q3, q4)
                    r'^[1-4]$',  # Simple 1, 2, 3, 4
                    r'^(?:first|second|third|fourth)[\s_-]*(?:quarter|qtr)$'  # Text formats
                ]
            },
            'day_of_week': {
                'name_patterns': [
                    r'\b(?:day[\s_-]*of[\s_-]*week|weekday|dow|day[\s_-]*name)\b',
                    r'\b(?:mon|tue|wed|thu|fri|sat|sun)(?:day|\.)?[\s_-]*(?:indicator|flag)?\b'
                ],
                'value_patterns': [
                    r'^(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)$',
                    r'^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)$',
                    r'^[0-6]$'  # 0-6 representation (often 0=Sunday or 0=Monday)
                ]
            },
            'day_of_month': {
                'name_patterns': [
                    r'\b(?:day|dy|dd|dom|day[\s_-]*of[\s_-]*month)\b',
                    r'\b(?:day)[\s_-]*(?:in|of)[\s_-]*(?:month|order|transaction|report)\b'
                ],
                'value_patterns': [
                    r'^(?:[1-9]|[12][0-9]|3[01])$',  # 1-31
                    r'^(?:0[1-9]|[12][0-9]|3[01])$'  # 01-31
                ]
            },
            'day_of_year': {
                'name_patterns': [
                    r'\b(?:day[\s_-]*of[\s_-]*year|doy|julian[\s_-]*day|ordinal[\s_-]*day)\b',
                    r'\b(?:yearly|annual)[\s_-]*(?:day|date)[\s_-]*(?:number|no)?\b'
                ],
                'value_patterns': [
                    r'^(?:[1-9]|[1-9][0-9]|[12][0-9]{2}|3[0-5][0-9]|36[0-6])$',  # 1-366
                    r'^(?:00[1-9]|0[1-9][0-9]|[12][0-9]{2}|3[0-5][0-9]|36[0-6])$'  # 001-366
                ]
            },
            'hour': {
                'name_patterns': [
                    r'\b(?:hour|hr|hrs|hh)\b',
                    r'\b(?:hour)[\s_-]*(?:of|in)[\s_-]*(?:day|order|transaction|report)\b'
                ],
                'value_patterns': [
                    r'^(?:[0-9]|1[0-9]|2[0-3])$',  # 0-23
                    r'^(?:0[0-9]|1[0-9]|2[0-3])$',  # 00-23
                    r'^(?:[1-9]|1[0-2])[\s_-]*(?:am|pm|AM|PM)$'  # 12-hour format
                ]
            },
            'minute': {
                'name_patterns': [
                    r'\b(?:minute|min|mm)\b',
                    r'\b(?:minute)[\s_-]*(?:of|in)[\s_-]*(?:hour|time|timestamp)\b'
                ],
                'value_patterns': [
                    r'^(?:[0-9]|[1-5][0-9])$',  # 0-59
                    r'^(?:0[0-9]|[1-5][0-9])$'  # 00-59
                ]
            },
            'second': {
                'name_patterns': [
                    r'\b(?:second|sec|ss)\b',
                    r'\b(?:second)[\s_-]*(?:of|in)[\s_-]*(?:minute|time|timestamp)\b'
                ],
                'value_patterns': [
                    r'^(?:[0-9]|[1-5][0-9])$',  # 0-59
                    r'^(?:0[0-9]|[1-5][0-9])$'  # 00-59
                ]
            },
            'period': {
                'name_patterns': [
                    r'\b(?:period|time[\s_-]*period|date[\s_-]*range|time[\s_-]*range|time[\s_-]*frame|window)\b',
                    r'\b(?:reporting|analysis|accounting|sales|fiscal)[\s_-]*(?:period|timeframe|time[\s_-]*frame)\b',
                    r'\b(?:from|start|begin|to|end|through)[\s_-]*(?:date|time|period)\b'
                ]
            },
            'season': {
                'name_patterns': [
                    r'\b(?:season|seasonal|quarter|period)\b',
                    r'\b(?:spring|summer|fall|autumn|winter)[\s_-]*(?:season|period|quarter)?\b',
                    r'\b(?:holiday|shopping|peak|off[\s_-]*peak)[\s_-]*(?:season|period|time)\b'
                ],
                'value_patterns': [
                    r'^(?:spring|summer|fall|autumn|winter)$',
                    r'^(?:q[1-4]|quarter[\s_-]*[1-4])$',
                    r'^(?:holiday|peak|off[\s_-]*peak|high|low)[\s_-]*(?:season)?$'
                ]
            }
        }
    
    def _initialize_financial_patterns(self):
        """Initialize financial patterns (expanded sales/revenue detection)"""
        return {
            'revenue': {
                'name_patterns': [
                    r'\b(?:revenue|sales|income|proceeds|turnover|earnings)\b',
                    r'\b(?:gross|net|total|monthly|quarterly|annual|daily)[\s_-]*(?:revenue|sales|income|proceeds)\b',
                    r'\b(?:revenue|sales)[\s_-]*(?:amount|figure|number|total|value|volume)\b'
                ]
            },
            'price': {
                'name_patterns': [
                    r'\b(?:price|cost|rate|charge|fee)\b',
                    r'\b(?:unit|per[\s_-]*item|single|individual)[\s_-]*(?:price|cost|rate|charge|fee)\b',
                    r'\b(?:price|cost)[\s_-]*(?:per|each|unit|item)\b'
                ]
            },
            'discount': {
                'name_patterns': [
                    r'\b(?:discount|reduction|markdown|savings|sale|promotion)\b',
                    r'\b(?:price|cost)[\s_-]*(?:discount|reduction|markdown|adjustment|deduction)\b',
                    r'\b(?:discount|promo|coupon|voucher)[\s_-]*(?:amount|value|percentage|rate|code)\b'
                ],
                'value_patterns': [
                    r'^-?\d+(?:\.\d+)?%?$',  # Numeric discount (may include % sign)
                    r'^-?\d+(?:\.\d+)?[\s_-]*(?:off|discount)$'  # Format like "25 off"
                ]
            },
            'tax': {
                'name_patterns': [
                    r'\b(?:tax|vat|gst|sales[\s_-]*tax|use[\s_-]*tax)\b',
                    r'\b(?:tax|vat|gst)[\s_-]*(?:amount|value|rate|percentage|total)\b',
                    r'\b(?:federal|state|local|city|county|provincial)[\s_-]*tax\b'
                ],
                'value_patterns': [
                    r'^\d+(?:\.\d+)?%?$'  # Tax rate (may include % sign)
                ]
            },
            'shipping': {
                'name_patterns': [
                    r'\b(?:shipping|freight|delivery|transport|postage|handling)\b',
                    r'\b(?:shipping|freight|delivery|transport)[\s_-]*(?:cost|charge|fee|expense|amount)\b',
                    r'\b(?:shipping|delivery)[\s_-]*(?:rate|price|cost|charge)\b'
                ]
            },
            'total': {
                'name_patterns': [
                    r'\b(?:total|sum|grand[\s_-]*total|final|overall)\b',
                    r'\b(?:total|final|overall)[\s_-]*(?:amount|cost|price|value|charge|fee)\b',
                    r'\b(?:order|invoice|purchase|transaction)[\s_-]*(?:total|amount|value|sum)\b'
                ]
            },
            'refund': {
                'name_patterns': [
                    r'\b(?:refund|return|reimbursement|chargeback|credit|money[\s_-]*back)\b',
                    r'\b(?:refund|return|reimbursement)[\s_-]*(?:amount|value|total|sum)\b',
                    r'\b(?:refunded|returned|credited|reimbursed)[\s_-]*(?:amount|value|total|sum)\b'
                ]
            },
            'cost': {
                'name_patterns': [
                    r'\b(?:cost|expense|expenditure|outlay|spend|spending)\b',
                    r'\b(?:cost|purchase|wholesale|acquisition)[\s_-]*(?:price|amount|value)\b',
                    r'\b(?:unit|per[\s_-]*item|manufacturing|production)[\s_-]*cost\b'
                ]
            },
            'margin': {
                'name_patterns': [
                    r'\b(?:margin|markup|profit[\s_-]*margin|contribution[\s_-]*margin)\b',
                    r'\b(?:gross|net|operating|profit)[\s_-]*margin\b',
                    r'\b(?:margin|markup)[\s_-]*(?:rate|percentage|amount|value)\b'
                ],
                'value_patterns': [
                    r'^\d+(?:\.\d+)?%?$'  # Margin rate (may include % sign)
                ]
            },
            'profit': {
                'name_patterns': [
                    r'\b(?:profit|earning|gain|surplus|advantage|yield|return)\b',
                    r'\b(?:gross|net|operating|pre[\s_-]*tax|after[\s_-]*tax)[\s_-]*(?:profit|earnings|income)\b',
                    r'\b(?:profit|earning)[\s_-]*(?:amount|value|total|sum)\b'
                ]
            },
            'currency': {
                'name_patterns': [
                    r'\b(?:currency|money|denomination|tender|legal[\s_-]*tender)\b',
                    r'\b(?:currency|monetary)[\s_-]*(?:code|unit|type|sign|symbol)\b',
                    r'\b(?:payment|transaction)[\s_-]*currency\b'
                ],
                'value_patterns': [
                    r'^(?:USD|EUR|GBP|JPY|CNY|AUD|CAD|CHF|INR|BRL)$',  # Common ISO codes
                    r'^[A-Z]{3}$',  # Any 3-letter code
                    r'^(?:\$|€|£|¥|₹|₽|₩|₿)$'  # Currency symbols
                ]
            }
        }
    
    def _initialize_product_patterns(self):
        """Initialize product patterns (expanded product detection)"""
        return {
            'product_id': {
                'name_patterns': [
                    r'\b(?:product|item|article|good|sku|upc|ean|isbn)[\s_-]*(?:id|code|number|no|identifier|key)\b',
                    r'\b(?:prod|prd|itm|art)[\s_-]*(?:id|code|num|no)\b',
                    r'\b(?:bar|qr)[\s_-]*code\b'
                ],
                'value_patterns': [
                    r'^[A-Z]{1,3}[-_]?\d{3,}$',  # ABC123, A-123, etc.
                    r'^\d{3,}[-_]?[A-Z]{1,3}$',  # 123ABC, 123-A, etc.
                    r'^(?:\d{8}|\d{12}|\d{13}|\d{14})$'  # UPC/EAN/ISBN formats
                ]
            },
            'product_name': {
                'name_patterns': [
                    r'\b(?:product|item|article|good|merchandise)[\s_-]*(?:name|title|label|description|desc)\b',
                    r'\b(?:prod|prd|itm|art)[\s_-]*(?:name|desc)\b',
                    r'\bdescription\b'
                ]
            },
            'product_category': {
                'name_patterns': [
                    r'\b(?:product|item)[\s_-]*(?:category|cat|type|group|class|classification|family)\b',
                    r'\b(?:category|cat|type|group|class|classification|family)[\s_-]*(?:of[\s_-]*(?:product|item))?\b',
                    r'\b(?:main|primary|secondary|sub)[\s_-]*(?:category|cat)\b'
                ],
                'value_patterns': [
                    r'^(?:electronics|clothing|food|beverage|furniture|home|garden|tools|automotive|beauty|health|toys|sports|books|music)$'
                ]
            },
            'product_brand': {
                'name_patterns': [
                    r'\b(?:brand|make|manufacturer|vendor|supplier|producer|creator)\b',
                    r'\b(?:product|item)[\s_-]*(?:brand|make|manufacturer|vendor)\b',
                    r'\b(?:brand|make|manufacturer)[\s_-]*(?:name|id|code)\b'
                ]
            },
           
            'product': {
                'name_patterns': [
                    r'\b(?:product|item|sku|merchandise|good|article|material)\b',
                    r'\b(?:product|item|article)[\s_-]*(?:id|code|number|no|identifier|key)\b'
                ],
                'id_patterns': [
                    r'^[A-Z]{1,3}[-_]?\d{3,}$',            # ABC123, A-123, etc.
                    r'^\d{3,}[-_]?[A-Z]{1,3}$',            # 123ABC, 123-A, etc.
                    r'^[A-Z0-9]{2,3}[-_]?\d{4,}$',         # AB1234, XYZ9876, etc.
                    r'^(?:[A-Z0-9]{2,5}[-_]){1,2}[A-Z0-9]{2,5}$'  # AB-12-XY, 123-ABC-45, etc.
                ]
            },
            'sales': {
                'name_patterns': [
                    r'\b(?:sales|revenue|amount|value|total|sum|selling)\b',
                    r'\b(?:total|gross)[\s_-]*(?:sales|revenue|income|amount|value|price)\b',
                    r'\b(?:extended|line)[\s_-]*(?:amount|price|total|value)\b'
                ]
            },
            'quantity': {
                'name_patterns': [
                    r'\b(?:quantity|qty|count|units|volume|number|pieces)\b',
                    r'\b(?:number|count)[\s_-]*(?:of|items|units|pieces)\b',
                    r'\b(?:item|unit|piece)[\s_-]*count\b'
                ],
                'value_patterns': [
                    r'^\d+$',  # Integer values
                ]
            },
            'price': {
                'name_patterns': [
                    r'\b(?:price|cost|rate|fee|charge)\b',
                    r'\b(?:unit|per[\s_-]*item)[\s_-]*(?:price|cost|value|amount)\b',
                    r'\b(?:price|cost)[\s_-]*(?:per|each|unit|item)\b'
                ]
            },
            'boolean': {
                'name_patterns': [
                    r'\b(?:is|has|should|can|will|flag|indicator|status|state)\b',
                    r'\b(?:is[\s_-]*(?:active|enabled|valid|canceled|approved|completed))\b',
                    r'\b(?:has[\s_-]*(?:approved|verified|validated|confirmed))\b'
                ],
                'value_patterns': [
                    r'^(?:0|1)$',
                    r'^(?:true|false)$',
                    r'^(?:yes|no)$',
                    r'^(?:y|n)$',
                    r'^(?:t|f)$'
                ]
            },
            'id': {
                'name_patterns': [
                    r'\b(?:id|identifier|key|code|number|num|no|#)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:id|identifier|key|code|number|num|no))\b',  # customer_id, order_id, etc.
                    r'\b(?:primary|foreign|unique)[\s_-]*(?:key|id|identifier)\b'
                ],
                'value_patterns': [
                    r'^\d{4,}$',  # 4+ digit numbers
                    r'^[A-Za-z]{1,3}[-_]?\d{3,}$',  # AB123, A-123
                    r'^\d{3,}[-_]?[A-Za-z]{1,3}$',  # 123AB, 123-A
                    r'^[A-Fa-f0-9]{8}-?[A-Fa-f0-9]{4}-?[A-Fa-f0-9]{4}-?[A-Fa-f0-9]{4}-?[A-Fa-f0-9]{12}$'  # UUID
                ]
            },
            'name': {
                'name_patterns': [
                    r'\b(?:name|title|label|caption|heading)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:name|title|label))\b',  # product_name, category_name, etc.
                    r'\b(?:full|display|short|long)[\s_-]*(?:name|title|label)\b'
                ]
            },
            'description': {
                'name_patterns': [
                    r'\b(?:description|desc|details|info|information|overview|summary)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:description|desc|details))\b',  # product_desc, item_details, etc.
                    r'\b(?:short|long|brief|full|detailed)[\s_-]*(?:description|desc|details)\b'
                ]
            },
            'category': {
                'name_patterns': [
                    r'\b(?:category|cat|type|group|class|classification|segment|tier|level)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:category|type|group|class))\b',  # product_category, item_type, etc.
                    r'\b(?:main|primary|secondary|sub)[\s_-]*(?:category|cat|type|group|class)\b'
                ]
            },
            'status': {
                'name_patterns': [
                    r'\b(?:status|state|condition|stage|phase|position|progress)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:status|state))\b',  # order_status, account_state, etc.
                    r'\b(?:current|previous|next|final)[\s_-]*(?:status|state|stage|phase)\b'
                ],
                'value_patterns': [
                    r'^(?:active|inactive|pending|completed|canceled|on[\s_-]*hold|new|processing|confirmed|shipped)$',
                    r'^(?:A|I|P|C|X|H|N|S)$'
                ]
            },
            'url': {
                'name_patterns': [
                    r'\b(?:url|uri|link|hyperlink|web[\s_-]*address|site)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:url|uri|link))\b',  # product_url, image_link, etc.
                    r'\b(?:http|https|ftp|website|webpage)[\s_-]*(?:link|url|address)?\b'
                ],
                'value_patterns': [
                    r'^(?:https?|ftp)://[^\s/$.?#].[^\s]*$',
                    r'^www\.[^\s/$.?#].[^\s]*$'
                ]
            },
            'email': {
                'name_patterns': [
                    r'\b(?:email|e[\s_-]*mail|mail)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:email|mail))\b',  # customer_email, contact_mail, etc.
                    r'\b(?:electronic|digital)[\s_-]*(?:mail|mailbox|address)\b'
                ],
                'value_patterns': [
                    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                ]
            },
            'phone': {
                'name_patterns': [
                    r'\b(?:phone|telephone|mobile|cell|contact)[\s_-]*(?:number|no|#)?\b',
                    r'\b(?:[a-z]+[\s_-]*(?:phone|telephone|mobile|cell))\b',  # customer_phone, contact_mobile, etc.
                    r'\b(?:work|home|business|fax)[\s_-]*(?:phone|telephone|number)\b'
                ],
                'value_patterns': [
                    r'^\+?[\d\s-\(\).]{7,}$'
                ]
            }
        }
    
    def _initialize_customer_patterns(self):
        """Initialize customer-related patterns"""
        return {
            'customer_id': {
                'name_patterns': [
                    r'\b(?:customer|client|buyer|account|contact)[\s_-]*(?:id|number|code|no|identifier|key)\b',
                    r'\b(?:cust|clnt|acct)[\s_-]*(?:id|code|num|no)\b',
                    r'\bcust[\s_-]*(?:#|no|num|code)\b'
                ],
                'value_patterns': [
                    r'^[A-Z]{1,3}-\d{4,10}$',
                    r'^CUST\d{4,}$',
                    r'^\d{5,10}$'  # Simple numeric customer IDs
                ]
            },
            'customer_name': {
                'name_patterns': [
                    r'\b(?:customer|client|buyer|account)[\s_-]*(?:name|title|fullname)\b',
                    r'\b(?:first|last|given|family|middle|full)[\s_-]*name\b',
                    r'\b(?:fname|lname|fullname)\b'
                ],
                'value_patterns': [
                    r'^[A-Z][a-z]+\s+[A-Z][a-z]+$',  # First Last format
                    r'^[A-Z][a-z]+,\s+[A-Z][a-z]+$'  # Last, First format
                ]
            },
            'customer_segment': {
                'name_patterns': [
                    r'\b(?:customer|client|buyer|account)[\s_-]*(?:segment|tier|group|level|type|category|cohort)\b',
                    r'\b(?:loyalty|vip|premium|gold|silver|bronze)[\s_-]*(?:tier|level|status|group)\b',
                    r'\b(?:segment|segmentation|cluster)[\s_-]*(?:id|code|name|group|value)?\b'
                ],
                'value_patterns': [
                    r'^(?:premium|standard|basic|gold|silver|bronze|platinum|vip|regular)$',
                    r'^tier[\s_-]?\d$',
                    r'^[A-D]$'  # Simple letter-based segments
                ]
            },
            'customer_acquisition': {
                'name_patterns': [
                    r'\b(?:customer|client|lead|prospect)[\s_-]*(?:acquisition|source|origin|channel)\b',
                    r'\b(?:acquisition|conversion|signup|registration|join|onboarding)[\s_-]*(?:source|channel|medium|campaign)\b',
                    r'\b(?:lead|referral|traffic)[\s_-]*source\b'
                ],
                'value_patterns': [
                    r'^(?:web|email|social|referral|organic|paid|direct|offline|store|call|event)$',
                    r'^(?:facebook|google|twitter|linkedin|instagram|tiktok|youtube)$'
                ]
            },
            'customer_ltv': {
                'name_patterns': [
                    r'\b(?:customer|client|buyer|account)[\s_-]*(?:ltv|lifetime[\s_-]*value|clv|value)\b',
                    r'\b(?:predicted|forecasted|estimated|projected)[\s_-]*(?:ltv|lifetime[\s_-]*value|clv)\b',
                    r'\b(?:ltv|clv|cltv)[\s_-]*(?:value|amount|score)?\b'
                ]
            },
            'customer_contact': {
                'name_patterns': [
                    r'\b(?:email|e-?mail|mail)[\s_-]*(?:address)?\b',
                    r'\b(?:phone|telephone|mobile|cell)[\s_-]*(?:number|no)?\b',
                    r'\bcontact[\s_-]*(?:details|info)\b'
                ],
                'value_patterns': {
                    'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                    'phone': r'^\+?[\d\s-\(\).]{7,}$'
                }
            },
            'customer_address': {
                'name_patterns': [
                    r'\b(?:billing|shipping|mailing|delivery|home|work|office)[\s_-]*address\b',
                    r'\b(?:street|avenue|road|boulevard|lane|plaza|building|apt|suite)[\s_-]*(?:address|number|no)?\b',
                    r'\b(?:address|addr)[\s_-]*(?:line[\s_-]*[1-3]|1|2|3)\b'
                ]
            },
            'customer_status': {
                'name_patterns': [
                    r'\b(?:customer|client|account)[\s_-]*(?:status|state|condition|standing)\b',
                    r'\b(?:active|inactive|suspended|pending|canceled|terminated|churned)[\s_-]*(?:status|flag|indicator)?\b',
                    r'\b(?:status|state)[\s_-]*(?:code|value|indicator)?\b'
                ],
                'value_patterns': [
                    r'^(?:active|inactive|pending|suspended|closed|terminated|on[\s_-]*hold|new|verified)$',
                    r'^(?:0|1|Y|N|A|I|P)$'  # Common status codes
                ]
            },
            'customer_age': {
                'name_patterns': [
                    r'\b(?:customer|client|user|member|account)[\s_-]*(?:age|years|yrs)\b',
                    r'\b(?:age|years|yrs)[\s_-]*(?:old)?\b',
                    r'\b(?:birth|dob|birth[\s_-]*date)[\s_-]*(?:date|day|year)?\b'
                ],
                'value_patterns': [
                    r'^(?:1[89]|[2-9][0-9])$',  # Ages 18-99
                    r'^\d{1,2}/\d{1,2}/\d{4}$',  # MM/DD/YYYY
                    r'^\d{4}-\d{2}-\d{2}$'  # YYYY-MM-DD
                ]
            },
            'customer_gender': {
                'name_patterns': [
                    r'\b(?:gender|sex|male|female)\b',
                    r'\b(?:customer|client|user|member|account)[\s_-]*(?:gender|sex)\b'
                ],
                'value_patterns': [
                    r'^(?:M|F|Male|Female|man|woman|non[\s_-]*binary|other)$',
                    r'^(?:m|f|male|female)$'
                ]
            },
            'customer_purchase_frequency': {
                'name_patterns': [
                    r'\b(?:customer|client|buyer|account)[\s_-]*(?:purchase|order|transaction)[\s_-]*(?:frequency|rate|count|number)\b',
                    r'\b(?:purchase|order|buy|transaction)[\s_-]*(?:frequency|rate|cadence|pattern|cycle)\b',
                    r'\b(?:frequency|times|count)[\s_-]*(?:of|per)[\s_-]*(?:purchase|order|transaction)\b'
                ]
            },
            'customer_churn_risk': {
                'name_patterns': [
                    r'\b(?:customer|client|account|churn|attrition)[\s_-]*(?:risk|score|probability|likelihood|potential)\b',
                    r'\b(?:churn|attrition|loss|cancellation|defection)[\s_-]*(?:risk|score|rate|prediction|probability)\b',
                    r'\b(?:risk|likelihood|probability)[\s_-]*(?:of|to)[\s_-]*(?:churn|cancel|leave|attrition)\b'
                ],
                'value_patterns': [
                    r'^(?:high|medium|low|H|M|L|critical|warning|safe)$',
                    r'^(?:0|[0-9]\.[0-9]+|[0-9]{1,2}|100)%?$'  # 0-100% or 0.0-1.0
                ]
            }
        }
    
    def _initialize_time_patterns(self):
        """Initialize time-related patterns"""
        return {
            'time_of_day': {
                'name_patterns': [
                    r'\b(?:time|hour|minute|second)[\s_-]*(?:of[\s_-]*day)?\b',
                    r'\b(?:timestamp|datetime|datetimeoffset)\b',
                    r'\b(?:clock|wall)[\s_-]*time\b'
                ],
                'value_patterns': [
                    r'^\d{1,2}:\d{2}(:\d{2})?([aApP][mM])?$',  # 3:30PM, 15:45, 9:20:15
                    r'^\d{1,2}[aApP][mM]$'  # 3PM, 11am
                ]
            },
            'duration': {
                'name_patterns': [
                    r'\b(?:duration|elapsed|length|span|interval|period)[\s_-]*(?:time|minutes|seconds|hours)?\b',
                    r'\b(?:time[\s_-]*spent|time[\s_-]*elapsed|time[\s_-]*taken)\b',
                    r'\b(?:session|call|visit|stay|usage)[\s_-]*(?:length|duration|time)\b'
                ],
                'value_patterns': [
                    r'^\d+:?\d*:?\d*$',  # 30, 1:30, 2:45:30
                    r'^\d+[\s_-]*(?:ms|sec|min|hr|day|wk|mo|yr)$'  # 30sec, 45min, 2hr
                ]
            },
            'delivery_time': {
                'name_patterns': [
                    r'\b(?:delivery|shipping|arrival|fulfillment|transit)[\s_-]*(?:time|date|timeframe|window|period|schedule)\b',
                    r'\b(?:estimated|actual|scheduled|promised|target)[\s_-]*(?:delivery|arrival|ship)[\s_-]*(?:date|time)\b',
                    r'\b(?:eta|etd|ata|atd)[\s_-]*(?:date|time)?\b'  # Estimated/Actual Time of Arrival/Departure
                ]
            },
            'lead_time': {
                'name_patterns': [
                    r'\b(?:lead|lag|processing|handling|turnaround|response)[\s_-]*time\b',
                    r'\b(?:time[\s_-]*to[\s_-]*(?:process|fulfill|complete|respond|ship|deliver))\b',
                    r'\b(?:sla|service[\s_-]*level[\s_-]*agreement)[\s_-]*(?:time|hours|days)?\b'
                ]
            },
            'time_zone': {
                'name_patterns': [
                    r'\b(?:time[\s_-]*zone|tz|timezone|time[\s_-]*offset|utc[\s_-]*offset)\b',
                    r'\b(?:local|server|system|user)[\s_-]*(?:time[\s_-]*zone|tz)\b'
                ],
                'value_patterns': [
                    r'^(?:UTC|GMT|EST|CST|MST|PST|EDT|CDT|MDT|PDT)[\+\-]?\d*$',
                    r'^[\+\-]\d{1,2}(?::\d{2})?$'  # +8, -5:30
                ]
            },
            'frequency': {
                'name_patterns': [
                    r'\b(?:frequency|interval|periodicity|recurrence|cycle|cadence)\b',
                    r'\b(?:daily|weekly|monthly|quarterly|yearly|annual)[\s_-]*(?:frequency|occurrence|schedule)?\b',
                    r'\b(?:times[\s_-]*per[\s_-]*(?:day|week|month|year|quarter))\b'
                ],
                'value_patterns': [
                    r'^(?:daily|weekly|biweekly|monthly|quarterly|annually|hourly|minutely)$',
                    r'^(?:every|once)[\s_-]+(?:day|week|month|year|hour|minute|second)$',
                    r'^\d+[\s_-]*(?:ms|sec|min|hr|day|wk|mo|yr)$'
                ]
            },
            'datetime_components': {
                'name_patterns': [
                    r'\b(?:year|yr|yyyy|fiscal[\s_-]*year|calendar[\s_-]*year)\b',
                    r'\b(?:month|mon|mm|quarter|qtr|season)\b',
                    r'\b(?:day|dy|dd|weekday|dow|dom)\b',
                    r'\b(?:hour|hr|minute|min|second|sec)\b'
                ],
                'value_patterns': {
                    'year': r'^(?:19|20)\d{2}$',  # 1900-2099
                    'month': r'^(?:0?[1-9]|1[0-2])$|^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$',
                    'day': r'^(?:0?[1-9]|[12][0-9]|3[01])$',
                    'hour': r'^(?:[01]?[0-9]|2[0-3])$',
                    'minute': r'^(?:[0-5]?[0-9])$',
                }
            },
            'week': {
                'name_patterns': [
                    r'\b(?:week|wk|week[\s_-]*number|week[\s_-]*no|weeknum)\b',
                    r'\b(?:iso|calendar|fiscal)[\s_-]*week\b',
                    r'\b(?:woy|wow|ww)[\s_-]*(?:number|no)?\b'  # Week of year, Week over week, Work week
                ],
                'value_patterns': [
                    r'^(?:[0-9]|[1-4][0-9]|5[0-3])$',  # 0-53
                    r'^W(?:[0-9]|[1-4][0-9]|5[0-3])$'  # W1-W53
                ]
            }
        }
    
    def _initialize_location_patterns(self):
        """Initialize geographic/location patterns"""
        return {
            'address': {
                'name_patterns': [
                    r'\b(?:address|addr|location|street|ave|blvd|road|rd)[\s_-]*(?:line)?[\s_-]*(?:1|2|3|one|two|three)?\b',
                    r'\b(?:billing|shipping|mailing|delivery|home|work|business)[\s_-]*address\b',
                    r'\b(?:street|building|location)[\s_-]*(?:address|number|name)\b'
                ]
            },
            'city': {
                'name_patterns': [
                    r'\b(?:city|town|municipality|borough|village|suburb|settlement|urban[\s_-]*area)\b',
                    r'\b(?:billing|shipping|mailing|delivery|home|work|business)[\s_-]*city\b',
                    r'\b(?:city|town)[\s_-]*(?:name|code)?\b'
                ]
            },
            'state_province': {
                'name_patterns': [
                    r'\b(?:state|province|county|region|district|territory|prefecture)\b',
                    r'\b(?:billing|shipping|mailing|delivery|home|work|business)[\s_-]*(?:state|province)\b',
                    r'\b(?:state|province)[\s_-]*(?:name|code|abbr|abbreviation)?\b'
                ],
                'value_patterns': [
                    r'^[A-Z]{2}$',  # US state codes: NY, CA, etc.
                    r'^(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)$'  # Specific US states
                ]
            },
            'postal_code': {
                'name_patterns': [
                    r'\b(?:zip|postal|post)[\s_-]*(?:code)?\b',
                    r'\b(?:billing|shipping|mailing|delivery|home|work|business)[\s_-]*(?:zip|postal|post)[\s_-]*(?:code)?\b'
                ],
                'value_patterns': [
                    r'^\d{5}(?:-\d{4})?$',  # US ZIP: 12345 or 12345-6789
                    r'^[A-Z]\d[A-Z][\s-]?\d[A-Z]\d$'  # Canadian postal code: A1A 1A1
                ]
            },
            'country': {
                'name_patterns': [
                    r'\b(?:country|nation|land|territory|commonwealth|republic|kingdom)\b',
                    r'\b(?:billing|shipping|mailing|delivery|home|work|business)[\s_-]*country\b',
                    r'\b(?:country|nation)[\s_-]*(?:name|code|iso)?\b'
                ],
                'value_patterns': [
                    r'^(?:US|USA|CA|CAN|UK|GB|AU|DE|FR|JP|CN|IN|BR)$',  # Common ISO codes
                    r'^[A-Z]{2}$',  # ISO 2-letter codes
                    r'^[A-Z]{3}$'   # ISO 3-letter codes
                ]
            },
            'region': {
                'name_patterns': [
                    r'\b(?:region|area|zone|district|sector|territory|jurisdiction)\b',
                    r'\b(?:sales|service|delivery|market)[\s_-]*(?:region|area|zone|territory)\b',
                    r'\b(?:region|zone)[\s_-]*(?:name|code|id)?\b'
                ]
            },
            'geo_coordinates': {
                'name_patterns': [
                    r'\b(?:latitude|lat|longitude|long|lon|lng|coords|coordinates|geo|gps|position)\b',
                    r'\b(?:lat|latitude)[\s_-]*(?:value|coordinate|position|degrees)?\b',
                    r'\b(?:lon|long|longitude)[\s_-]*(?:value|coordinate|position|degrees)?\b'
                ],
                'value_patterns': {
                    'latitude': r'^-?(?:90|[1-8]?[0-9](?:\.\d+)?)$',
                    'longitude': r'^-?(?:180|1[0-7][0-9]|[1-9]?[0-9](?:\.\d+)?)$'
                }
            },
            'store_location': {
                'name_patterns': [
                    r'\b(?:store|shop|outlet|branch|location|site|dealer|franchise)[\s_-]*(?:id|code|number|location|address)?\b',
                    r'\b(?:retail|warehouse|distribution|pickup|collection)[\s_-]*(?:location|site|center|facility)\b'
                ],
                'value_patterns': [
                    r'^[A-Z]{1,2}[-_]?\d{2,4}$',  # Store codes like NY-123, TX12
                    r'^ST\d{3,}$',  # ST123, ST4567
                    r'^\d{3,5}$'    # Simple numeric store IDs
                ]
            }
        }
    
    def _initialize_marketing_patterns(self):
        """Initialize marketing and campaign patterns"""
        return {
            'campaign': {
                'name_patterns': [
                    r'\b(?:campaign|promotion|promo|initiative|drive|marketing)[\s_-]*(?:id|code|name|number|no|identifier)?\b',
                    r'\b(?:advertisement|ad|advert)[\s_-]*(?:campaign|id|code|name|number)?\b',
                    r'\b(?:camp|cmpgn)[\s_-]*(?:id|code|num|no)?\b'
                ]
            },
            'channel': {
                'name_patterns': [
                    r'\b(?:channel|medium|platform|source|touchpoint|outlet)\b',
                    r'\b(?:marketing|sales|distribution|communication|acquisition)[\s_-]*(?:channel|source|medium)\b',
                    r'\b(?:utm|tracking)[\s_-]*(?:source|medium|channel)\b'
                ],
                'value_patterns': [
                    r'^(?:web|email|social|print|tv|radio|direct|mail|store|online|offline|mobile|app|referral)$',
                    r'^(?:facebook|twitter|instagram|linkedin|youtube|google|pinterest|tiktok)$'
                ]
            },
            'promotion': {
                'name_patterns': [
                    r'\b(?:promotion|promo|offer|deal|discount|special)[\s_-]*(?:id|code|name|type)?\b',
                    r'\b(?:coupon|voucher|rebate)[\s_-]*(?:code|id|number|amount|value|percent)?\b',
                    r'\b(?:promo|discount)[\s_-]*(?:rate|amount|value|percentage|level|tier)?\b'
                ]
            },
            'segment': {
                'name_patterns': [
                    r'\b(?:segment|segmentation|audience|cohort|group|cluster|category)\b',
                    r'\b(?:market|customer|user|buyer|client|visitor)[\s_-]*(?:segment|group|cluster|category|cohort)\b',
                    r'\b(?:demographic|psychographic|behavioral)[\s_-]*(?:segment|group|cluster|category)?\b'
                ]
            },
            'conversion': {
                'name_patterns': [
                    r'\b(?:conversion|click|action|engagement|interaction|event)[\s_-]*(?:rate|percentage|count|number|value)?\b',
                    r'\b(?:bounce|exit|abandonment)[\s_-]*rate\b',
                    r'\b(?:ctr|cvr|cpa|cpc|cpm)[\s_-]*(?:value|rate|amount)?\b'  # Click-through rate, Conversion rate, etc.
                ]
            },
            'attribution': {
                'name_patterns': [
                    r'\b(?:attribution|contribution|credit|source|origin|influence)\b',
                    r'\b(?:first|last|multi|linear|time|position|touch)[\s_-]*(?:touch|click|interaction|attribution|credit)\b',
                    r'\b(?:attribution|contribution)[\s_-]*(?:model|method|approach|algorithm|rule|logic)\b'
                ],
                'value_patterns': [
                    r'^(?:first|last|linear|time|position|multi|data)[\s_-]*(?:touch|click|interaction|attribution)$',
                    r'^(?:direct|organic|referral|social|email|paid|affiliate|partner)$'
                ]
            },
            'marketing_cost': {
                'name_patterns': [
                    r'\b(?:marketing|advertising|promotion|campaign)[\s_-]*(?:cost|expense|spend|budget|investment)\b',
                    r'\b(?:ad|campaign|promotion)[\s_-]*(?:spend|cost|expense|budget)\b',
                    r'\b(?:cac|cpa|cpc|cpm|cpp)[\s_-]*(?:cost|value|amount|rate)?\b'  # Cost per acquisition, Cost per click, etc.
                ]
            },
            'marketing_roi': {
                'name_patterns': [
                    r'\b(?:marketing|advertising|promotion|campaign)[\s_-]*(?:roi|return|roas|performance|efficiency)\b',
                    r'\b(?:roi|roas|romi|return)[\s_-]*(?:on|of)[\s_-]*(?:marketing|advertising|ad|campaign|promotion)[\s_-]*(?:spend|investment|expense)?\b',
                    r'\b(?:campaign|ad|promotion)[\s_-]*(?:roi|return|performance|results|success|effectiveness)\b'
                ],
                'value_patterns': [
                    r'^-?\d+(?:\.\d+)?%?$',  # Numeric ROI (possibly with percentage sign)
                    r'^-?\d+(?:\.\d+)?[xX]$'  # ROI in format like "3.5x"
                ]
            }
        }
    
    def _initialize_operational_patterns(self):
        """Initialize operational/fulfillment patterns"""
        return {
            'order_status': {
                'name_patterns': [
                    r'\b(?:order|shipment|delivery|fulfillment|transaction)[\s_-]*(?:status|state|condition|stage|phase)\b',
                    r'\b(?:current|latest|updated|tracking)[\s_-]*(?:status|state|condition)\b',
                    r'\b(?:status|state)[\s_-]*(?:code|value|indicator|flag)?\b'
                ],
                'value_patterns': [
                    r'^(?:new|pending|processing|shipped|delivered|completed|cancelled|returned|on[-\s_]*hold|back[-\s_]*ordered)$',
                    r'^(?:N|P|S|D|C|X|R|H|B)$'  # Status codes
                ]
            },
            'shipping_method': {
                'name_patterns': [
                    r'\b(?:shipping|delivery|transport|carrier|shipment|fulfillment)[\s_-]*(?:method|type|mode|service|option|provider|carrier|company)?\b',
                    r'\b(?:express|standard|overnight|priority|ground|air|freight)[\s_-]*(?:shipping|delivery|service)?\b',
                    r'\b(?:ship|delivery)[\s_-]*(?:via|by|through|method|type|mode)\b'
                ],
                'value_patterns': [
                    r'^(?:standard|express|overnight|priority|ground|air|2[-\s_]*day|next[-\s_]*day)$',
                    r'^(?:fedex|ups|usps|dhl|amazon|royal[-\s_]*mail|canada[-\s_]*post)$'
                ]
            },
            'tracking_id': {
                'name_patterns': [
                    r'\b(?:tracking|shipment|package|parcel|delivery)[\s_-]*(?:id|number|code|identifier|reference|no)\b',
                    r'\b(?:track|trace)[\s_-]*(?:id|no|number|code)\b',
                    r'\b(?:waybill|airway[-\s_]*bill|bill[-\s_]*of[-\s_]*lading)[\s_-]*(?:number|no|id|code)?\b'
                ],
                'value_patterns': [
                    r'^[0-9]{8,15}$',  # Basic numeric tracking
                    r'^[A-Z]{2}[0-9]{9}[A-Z]{2}$',  # USPS format
                    r'^1Z[A-Z0-9]{16}$'  # UPS format
                ]
            },
            'fulfillment_center': {
                'name_patterns': [
                    r'\b(?:fulfillment|distribution|warehouse|storage|inventory|logistics)[\s_-]*(?:center|facility|location|building|site|hub|depot)\b',
                    r'\b(?:fc|dc|wh)[\s_-]*(?:id|code|number|name|location)?\b',
                    r'\b(?:shipping|fulfillment|shipping)[\s_-]*(?:from|origin|source|location)\b'
                ]
            },
            'inventory_status': {
                'name_patterns': [
                    r'\b(?:inventory|stock|supply|availability|quantity)[\s_-]*(?:status|level|state|condition|position)\b',
                    r'\b(?:in[-\s_]*stock|out[-\s_]*of[-\s_]*stock|available|unavailable|backorder)[\s_-]*(?:status|flag|indicator)?\b',
                    r'\b(?:stock|inventory)[\s_-]*(?:on[-\s_]*hand|available|reserved|allocated|committed)\b'
                ],
                'value_patterns': [
                    r'^(?:in[-\s_]*stock|out[-\s_]*of[-\s_]*stock|low[-\s_]*stock|available|unavailable|backorder)$',
                    r'^(?:1|0|Y|N|A|U|L|B)$'  # Status codes (1/0, Yes/No, etc.)
                ]
            },
            'batch_number': {
                'name_patterns': [
                    r'\b(?:batch|lot|production|manufacturing)[\s_-]*(?:number|no|id|code|identifier)\b',
                    r'\b(?:batch|lot)[\s_-]*(?:qty|quantity|volume|amount|count)?\b',
                    r'\b(?:production|process|manufacturing)[\s_-]*(?:run|sequence|series)[\s_-]*(?:id|no|number)?\b'
                ],
                'value_patterns': [
                    r'^[A-Z]{1,3}[0-9]{4,8}$',  # BA12345
                    r'^[0-9]{4,8}[A-Z]{1,3}$',  # 12345BA
                    r'^(?:B|L|P)-[0-9]{4,8}$'  # B-12345
                ]
            },
            'order_id': {
                'name_patterns': [
                    r'\b(?:order|purchase|transaction|invoice|sales|checkout)[\s_-]*(?:id|number|no|code|identifier|reference)\b',
                    r'\b(?:ord|po|pur|inv)[\s_-]*(?:id|no|number|code|#)\b',
                    r'\b(?:confirmation|receipt)[\s_-]*(?:number|no|id|code)\b'
                ],
                'value_patterns': [
                    r'^ORD-?\d{5,10}$',  # ORD12345, ORD-12345
                    r'^PO-?\d{5,10}$',   # PO12345, PO-12345
                    r'^\d{5,10}$'        # Simple numeric order IDs
                ]
            },
            'order_line': {
                'name_patterns': [
                    r'\b(?:order|line|item)[\s_-]*(?:line|item|position|number|sequence|sequence[\s_-]*number)\b',
                    r'\b(?:line|item)[\s_-]*(?:in|on|of)[\s_-]*(?:order|invoice|receipt)\b',
                    r'\b(?:item|position|line)[\s_-]*(?:#|no|number)\b'
                ],
                'value_patterns': [
                    r'^\d{1,3}$',  # Simple line numbers (1, 2, 3, etc.)
                    r'^\d{1,3}\.\d{1,3}$'  # Hierarchical line numbers (1.1, 1.2, etc.)
                ]
            }
        }
    
    def _initialize_temporal_patterns(self):
        """Initialize temporal patterns (expanded date/time detection)"""
        return {
            'year': {
                'name_patterns': [
                    r'\b(?:year|yr|yyyy|fiscal[\s_-]*year|fy|annual|calendar[\s_-]*year)\b',
                    r'\b(?:year|yr)[\s_-]*(?:of|in)[\s_-]*(?:order|purchase|transaction|sale|report)\b'
                ],
                'value_patterns': [
                    r'^(?:19|20)\d{2}$',  # 1900-2099
                    r'^\d{2}$',  # 2-digit years (22, 23, etc.)
                    r'^FY\d{2}(?:\d{2})?$'  # Fiscal year formats (FY22, FY2022)
                ]
            },
            'month': {
                'name_patterns': [
                    r'\b(?:month|mon|mm|mo|mnth)\b',
                    r'\b(?:month|mon)[\s_-]*(?:of|in)[\s_-]*(?:order|purchase|transaction|sale|year|report)\b',
                    r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(?:uary|ruary|ch|il|ust|tember|ober|ember)?\b'
                ],
                'value_patterns': [
                    r'^(?:0?[1-9]|1[0-2])$',  # 1-12 (or 01-12)
                    r'^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$',  # Three-letter abbreviations
                    r'^(?:January|February|March|April|May|June|July|August|September|October|November|December)$'  # Full names
                ]
            },
            'quarter': {
                'name_patterns': [
                    r'\b(?:quarter|qtr|q[1-4]|fiscal[\s_-]*quarter)\b',
                    r'\b(?:quarter|qtr)[\s_-]*(?:[1-4]|one|two|three|four)\b',
                    r'\b(?:q|quarter)[\s_-]*(?:of|in)[\s_-]*(?:year|fiscal|fy)\b'
                ],
                'value_patterns': [
                    r'^[Qq][1-4]$',  # Q1, Q2, Q3, Q4 (or q1, q2, q3, q4)
                    r'^[1-4]$',  # Simple 1, 2, 3, 4
                    r'^(?:first|second|third|fourth)[\s_-]*(?:quarter|qtr)$'  # Text formats
                ]
            },
            'day_of_week': {
                'name_patterns': [
                    r'\b(?:day[\s_-]*of[\s_-]*week|weekday|dow|day[\s_-]*name)\b',
                    r'\b(?:mon|tue|wed|thu|fri|sat|sun)(?:day|\.)?[\s_-]*(?:indicator|flag)?\b'
                ],
                'value_patterns': [
                    r'^(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)$',
                    r'^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)$',
                    r'^[0-6]$'  # 0-6 representation (often 0=Sunday or 0=Monday)
                ]
            },
            'day_of_month': {
                'name_patterns': [
                    r'\b(?:day|dy|dd|dom|day[\s_-]*of[\s_-]*month)\b',
                    r'\b(?:day)[\s_-]*(?:in|of)[\s_-]*(?:month|order|transaction|report)\b'
                ],
                'value_patterns': [
                    r'^(?:[1-9]|[12][0-9]|3[01])$',  # 1-31
                    r'^(?:0[1-9]|[12][0-9]|3[01])$'  # 01-31
                ]
            },
            'day_of_year': {
                'name_patterns': [
                    r'\b(?:day[\s_-]*of[\s_-]*year|doy|julian[\s_-]*day|ordinal[\s_-]*day)\b',
                    r'\b(?:yearly|annual)[\s_-]*(?:day|date)[\s_-]*(?:number|no)?\b'
                ],
                'value_patterns': [
                    r'^(?:[1-9]|[1-9][0-9]|[12][0-9]{2}|3[0-5][0-9]|36[0-6])$',  # 1-366
                    r'^(?:00[1-9]|0[1-9][0-9]|[12][0-9]{2}|3[0-5][0-9]|36[0-6])$'  # 001-366
                ]
            },
            'hour': {
                'name_patterns': [
                    r'\b(?:hour|hr|hrs|hh)\b',
                    r'\b(?:hour)[\s_-]*(?:of|in)[\s_-]*(?:day|order|transaction|report)\b'
                ],
                'value_patterns': [
                    r'^(?:[0-9]|1[0-9]|2[0-3])$',  # 0-23
                    r'^(?:0[0-9]|1[0-9]|2[0-3])$',  # 00-23
                    r'^(?:[1-9]|1[0-2])[\s_-]*(?:am|pm|AM|PM)$'  # 12-hour format
                ]
            },
            'minute': {
                'name_patterns': [
                    r'\b(?:minute|min|mm)\b',
                    r'\b(?:minute)[\s_-]*(?:of|in)[\s_-]*(?:hour|time|timestamp)\b'
                ],
                'value_patterns': [
                    r'^(?:[0-9]|[1-5][0-9])$',  # 0-59
                    r'^(?:0[0-9]|[1-5][0-9])$'  # 00-59
                ]
            },
            'second': {
                'name_patterns': [
                    r'\b(?:second|sec|ss)\b',
                    r'\b(?:second)[\s_-]*(?:of|in)[\s_-]*(?:minute|time|timestamp)\b'
                ],
                'value_patterns': [
                    r'^(?:[0-9]|[1-5][0-9])$',  # 0-59
                    r'^(?:0[0-9]|[1-5][0-9])$'  # 00-59
                ]
            },
            'period': {
                'name_patterns': [
                    r'\b(?:period|time[\s_-]*period|date[\s_-]*range|time[\s_-]*range|time[\s_-]*frame|window)\b',
                    r'\b(?:reporting|analysis|accounting|sales|fiscal)[\s_-]*(?:period|timeframe|time[\s_-]*frame)\b',
                    r'\b(?:from|start|begin|to|end|through)[\s_-]*(?:date|time|period)\b'
                ]
            },
            'season': {
                'name_patterns': [
                    r'\b(?:season|seasonal|quarter|period)\b',
                    r'\b(?:spring|summer|fall|autumn|winter)[\s_-]*(?:season|period|quarter)?\b',
                    r'\b(?:holiday|shopping|peak|off[\s_-]*peak)[\s_-]*(?:season|period|time)\b'
                ],
                'value_patterns': [
                    r'^(?:spring|summer|fall|autumn|winter)$',
                    r'^(?:q[1-4]|quarter[\s_-]*[1-4])$',
                    r'^(?:holiday|peak|off[\s_-]*peak|high|low)[\s_-]*(?:season)?$'
                ]
            }
        }
    
    def _initialize_financial_patterns(self):
        """Initialize financial patterns (expanded sales/revenue detection)"""
        return {
            'revenue': {
                'name_patterns': [
                    r'\b(?:revenue|sales|income|proceeds|turnover|earnings)\b',
                    r'\b(?:gross|net|total|monthly|quarterly|annual|daily)[\s_-]*(?:revenue|sales|income|proceeds)\b',
                    r'\b(?:revenue|sales)[\s_-]*(?:amount|figure|number|total|value|volume)\b'
                ]
            },
            'price': {
                'name_patterns': [
                    r'\b(?:price|cost|rate|charge|fee)\b',
                    r'\b(?:unit|per[\s_-]*item|single|individual)[\s_-]*(?:price|cost|rate|charge|fee)\b',
                    r'\b(?:price|cost)[\s_-]*(?:per|each|unit|item)\b'
                ]
            },
            'discount': {
                'name_patterns': [
                    r'\b(?:discount|reduction|markdown|savings|sale|promotion)\b',
                    r'\b(?:price|cost)[\s_-]*(?:discount|reduction|markdown|adjustment|deduction)\b',
                    r'\b(?:discount|promo|coupon|voucher)[\s_-]*(?:amount|value|percentage|rate|code)\b'
                ],
                'value_patterns': [
                    r'^-?\d+(?:\.\d+)?%?$',  # Numeric discount (may include % sign)
                    r'^-?\d+(?:\.\d+)?[\s_-]*(?:off|discount)$'  # Format like "25 off"
                ]
            },
            'tax': {
                'name_patterns': [
                    r'\b(?:tax|vat|gst|sales[\s_-]*tax|use[\s_-]*tax)\b',
                    r'\b(?:tax|vat|gst)[\s_-]*(?:amount|value|rate|percentage|total)\b',
                    r'\b(?:federal|state|local|city|county|provincial)[\s_-]*tax\b'
                ],
                'value_patterns': [
                    r'^\d+(?:\.\d+)?%?$'  # Tax rate (may include % sign)
                ]
            },
            'shipping': {
                'name_patterns': [
                    r'\b(?:shipping|freight|delivery|transport|postage|handling)\b',
                    r'\b(?:shipping|freight|delivery|transport)[\s_-]*(?:cost|charge|fee|expense|amount)\b',
                    r'\b(?:shipping|delivery)[\s_-]*(?:rate|price|cost|charge)\b'
                ]
            },
            'total': {
                'name_patterns': [
                    r'\b(?:total|sum|grand[\s_-]*total|final|overall)\b',
                    r'\b(?:total|final|overall)[\s_-]*(?:amount|cost|price|value|charge|fee)\b',
                    r'\b(?:order|invoice|purchase|transaction)[\s_-]*(?:total|amount|value|sum)\b'
                ]
            },
            'refund': {
                'name_patterns': [
                    r'\b(?:refund|return|reimbursement|chargeback|credit|money[\s_-]*back)\b',
                    r'\b(?:refund|return|reimbursement)[\s_-]*(?:amount|value|total|sum)\b',
                    r'\b(?:refunded|returned|credited|reimbursed)[\s_-]*(?:amount|value|total|sum)\b'
                ]
            },
            'cost': {
                'name_patterns': [
                    r'\b(?:cost|expense|expenditure|outlay|spend|spending)\b',
                    r'\b(?:cost|purchase|wholesale|acquisition)[\s_-]*(?:price|amount|value)\b',
                    r'\b(?:unit|per[\s_-]*item|manufacturing|production)[\s_-]*cost\b'
                ]
            },
            'margin': {
                'name_patterns': [
                    r'\b(?:margin|markup|profit[\s_-]*margin|contribution[\s_-]*margin)\b',
                    r'\b(?:gross|net|operating|profit)[\s_-]*margin\b',
                    r'\b(?:margin|markup)[\s_-]*(?:rate|percentage|amount|value)\b'
                ],
                'value_patterns': [
                    r'^\d+(?:\.\d+)?%?$'  # Margin rate (may include % sign)
                ]
            },
            'profit': {
                'name_patterns': [
                    r'\b(?:profit|earning|gain|surplus|advantage|yield|return)\b',
                    r'\b(?:gross|net|operating|pre[\s_-]*tax|after[\s_-]*tax)[\s_-]*(?:profit|earnings|income)\b',
                    r'\b(?:profit|earning)[\s_-]*(?:amount|value|total|sum)\b'
                ]
            },
            'currency': {
                'name_patterns': [
                    r'\b(?:currency|money|denomination|tender|legal[\s_-]*tender)\b',
                    r'\b(?:currency|monetary)[\s_-]*(?:code|unit|type|sign|symbol)\b',
                    r'\b(?:payment|transaction)[\s_-]*currency\b'
                ],
                'value_patterns': [
                    r'^(?:USD|EUR|GBP|JPY|CNY|AUD|CAD|CHF|INR|BRL)$',  # Common ISO codes
                    r'^[A-Z]{3}$',  # Any 3-letter code
                    r'^(?:\$|€|£|¥|₹|₽|₩|₿)$'  # Currency symbols
                ]
            }
        }
    
    def _initialize_product_patterns(self):
        """Initialize product patterns (expanded product detection)"""
        return {
            'product_id': {
                'name_patterns': [
                    r'\b(?:product|item|article|good|sku|upc|ean|isbn)[\s_-]*(?:id|code|number|no|identifier|key)\b',
                    r'\b(?:prod|prd|itm|art)[\s_-]*(?:id|code|num|no)\b',
                    r'\b(?:bar|qr)[\s_-]*code\b'
                ],
                'value_patterns': [
                    r'^[A-Z]{1,3}[-_]?\d{3,}$',  # ABC123, A-123, etc.
                    r'^\d{3,}[-_]?[A-Z]{1,3}$',  # 123ABC, 123-A, etc.
                    r'^(?:\d{8}|\d{12}|\d{13}|\d{14})$'  # UPC/EAN/ISBN formats
                ]
            },
            'product_name': {
                'name_patterns': [
                    r'\b(?:product|item|article|good|merchandise)[\s_-]*(?:name|title|label|description|desc)\b',
                    r'\b(?:prod|prd|itm|art)[\s_-]*(?:name|desc)\b',
                    r'\bdescription\b'
                ]
            },
            'product_category': {
                'name_patterns': [
                    r'\b(?:product|item)[\s_-]*(?:category|cat|type|group|class|classification|family)\b',
                    r'\b(?:category|cat|type|group|class|classification|family)[\s_-]*(?:of[\s_-]*(?:product|item))?\b',
                    r'\b(?:main|primary|secondary|sub)[\s_-]*(?:category|cat)\b'
                ],
                'value_patterns': [
                    r'^(?:electronics|clothing|food|beverage|furniture|home|garden|tools|automotive|beauty|health|toys|sports|books|music)$'
                ]
            },
            'product_brand': {
                'name_patterns': [
                    r'\b(?:brand|make|manufacturer|vendor|supplier|producer|creator)\b',
                    r'\b(?:product|item)[\s_-]*(?:brand|make|manufacturer|vendor)\b',
                    r'\b(?:brand|make|manufacturer)[\s_-]*(?:name|id|code)\b'
                ]
            },
           
            'product': {
                'name_patterns': [
                    r'\b(?:product|item|sku|merchandise|good|article|material)\b',
                    r'\b(?:product|item|article)[\s_-]*(?:id|code|number|no|identifier|key)\b'
                ],
                'id_patterns': [
                    r'^[A-Z]{1,3}[-_]?\d{3,}$',            # ABC123, A-123, etc.
                    r'^\d{3,}[-_]?[A-Z]{1,3}$',            # 123ABC, 123-A, etc.
                    r'^[A-Z0-9]{2,3}[-_]?\d{4,}$',         # AB1234, XYZ9876, etc.
                    r'^(?:[A-Z0-9]{2,5}[-_]){1,2}[A-Z0-9]{2,5}$'  # AB-12-XY, 123-ABC-45, etc.
                ]
            },
            'sales': {
                'name_patterns': [
                    r'\b(?:sales|revenue|amount|value|total|sum|selling)\b',
                    r'\b(?:total|gross)[\s_-]*(?:sales|revenue|income|amount|value|price)\b',
                    r'\b(?:extended|line)[\s_-]*(?:amount|price|total|value)\b'
                ]
            },
            'quantity': {
                'name_patterns': [
                    r'\b(?:quantity|qty|count|units|volume|number|pieces)\b',
                    r'\b(?:number|count)[\s_-]*(?:of|items|units|pieces)\b',
                    r'\b(?:item|unit|piece)[\s_-]*count\b'
                ],
                'value_patterns': [
                    r'^\d+$',  # Integer values
                ]
            },
            'price': {
                'name_patterns': [
                    r'\b(?:price|cost|rate|fee|charge)\b',
                    r'\b(?:unit|per[\s_-]*item)[\s_-]*(?:price|cost|value|amount)\b',
                    r'\b(?:price|cost)[\s_-]*(?:per|each|unit|item)\b'
                ]
            },
            'boolean': {
                'name_patterns': [
                    r'\b(?:is|has|should|can|will|flag|indicator|status|state)\b',
                    r'\b(?:is[\s_-]*(?:active|enabled|valid|canceled|approved|completed))\b',
                    r'\b(?:has[\s_-]*(?:approved|verified|validated|confirmed))\b'
                ],
                'value_patterns': [
                    r'^(?:0|1)$',
                    r'^(?:true|false)$',
                    r'^(?:yes|no)$',
                    r'^(?:y|n)$',
                    r'^(?:t|f)$'
                ]
            },
            'id': {
                'name_patterns': [
                    r'\b(?:id|identifier|key|code|number|num|no|#)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:id|identifier|key|code|number|num|no))\b',  # customer_id, order_id, etc.
                    r'\b(?:primary|foreign|unique)[\s_-]*(?:key|id|identifier)\b'
                ],
                'value_patterns': [
                    r'^\d{4,}$',  # 4+ digit numbers
                    r'^[A-Za-z]{1,3}[-_]?\d{3,}$',  # AB123, A-123
                    r'^\d{3,}[-_]?[A-Za-z]{1,3}$',  # 123AB, 123-A
                    r'^[A-Fa-f0-9]{8}-?[A-Fa-f0-9]{4}-?[A-Fa-f0-9]{4}-?[A-Fa-f0-9]{4}-?[A-Fa-f0-9]{12}$'  # UUID
                ]
            },
            'name': {
                'name_patterns': [
                    r'\b(?:name|title|label|caption|heading)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:name|title|label))\b',  # product_name, category_name, etc.
                    r'\b(?:full|display|short|long)[\s_-]*(?:name|title|label)\b'
                ]
            },
            'description': {
                'name_patterns': [
                    r'\b(?:description|desc|details|info|information|overview|summary)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:description|desc|details))\b',  # product_desc, item_details, etc.
                    r'\b(?:short|long|brief|full|detailed)[\s_-]*(?:description|desc|details)\b'
                ]
            },
            'category': {
                'name_patterns': [
                    r'\b(?:category|cat|type|group|class|classification|segment|tier|level)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:category|type|group|class))\b',  # product_category, item_type, etc.
                    r'\b(?:main|primary|secondary|sub)[\s_-]*(?:category|cat|type|group|class)\b'
                ]
            },
            'status': {
                'name_patterns': [
                    r'\b(?:status|state|condition|stage|phase|position|progress)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:status|state))\b',  # order_status, account_state, etc.
                    r'\b(?:current|previous|next|final)[\s_-]*(?:status|state|stage|phase)\b'
                ],
                'value_patterns': [
                    r'^(?:active|inactive|pending|completed|canceled|on[\s_-]*hold|new|processing|confirmed|shipped)$',
                    r'^(?:A|I|P|C|X|H|N|S)$'
                ]
            },
            'url': {
                'name_patterns': [
                    r'\b(?:url|uri|link|hyperlink|web[\s_-]*address|site)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:url|uri|link))\b',  # product_url, image_link, etc.
                    r'\b(?:http|https|ftp|website|webpage)[\s_-]*(?:link|url|address)?\b'
                ],
                'value_patterns': [
                    r'^(?:https?|ftp)://[^\s/$.?#].[^\s]*$',
                    r'^www\.[^\s/$.?#].[^\s]*$'
                ]
            },
            'email': {
                'name_patterns': [
                    r'\b(?:email|e[\s_-]*mail|mail)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:email|mail))\b',  # customer_email, contact_mail, etc.
                    r'\b(?:electronic|digital)[\s_-]*(?:mail|mailbox|address)\b'
                ],
                'value_patterns': [
                    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                ]
            },
            'phone': {
                'name_patterns': [
                    r'\b(?:phone|telephone|mobile|cell|contact)[\s_-]*(?:number|no|#)?\b',
                    r'\b(?:[a-z]+[\s_-]*(?:phone|telephone|mobile|cell))\b',  # customer_phone, contact_mobile, etc.
                    r'\b(?:work|home|business|fax)[\s_-]*(?:phone|telephone|number)\b'
                ],
                'value_patterns': [
                    r'^\+?[\d\s-\(\).]{7,}$'
                ]
            }
        }
    
    def _initialize_customer_patterns(self):
        """Initialize customer-related patterns"""
        return {
            'customer_id': {
                'name_patterns': [
                    r'\b(?:customer|client|buyer|account|contact)[\s_-]*(?:id|number|code|no|identifier|key)\b',
                    r'\b(?:cust|clnt|acct)[\s_-]*(?:id|code|num|no)\b',
                    r'\bcust[\s_-]*(?:#|no|num|code)\b'
                ],
                'value_patterns': [
                    r'^[A-Z]{1,3}-\d{4,10}$',
                    r'^CUST\d{4,}$',
                    r'^\d{5,10}$'  # Simple numeric customer IDs
                ]
            },
            'customer_name': {
                'name_patterns': [
                    r'\b(?:customer|client|buyer|account)[\s_-]*(?:name|title|fullname)\b',
                    r'\b(?:first|last|given|family|middle|full)[\s_-]*name\b',
                    r'\b(?:fname|lname|fullname)\b'
                ],
                'value_patterns': [
                    r'^[A-Z][a-z]+\s+[A-Z][a-z]+$',  # First Last format
                    r'^[A-Z][a-z]+,\s+[A-Z][a-z]+$'  # Last, First format
                ]
            },
            'customer_segment': {
                'name_patterns': [
                    r'\b(?:customer|client|buyer|account)[\s_-]*(?:segment|tier|group|level|type|category|cohort)\b',
                    r'\b(?:loyalty|vip|premium|gold|silver|bronze)[\s_-]*(?:tier|level|status|group)\b',
                    r'\b(?:segment|segmentation|cluster)[\s_-]*(?:id|code|name|group|value)?\b'
                ],
                'value_patterns': [
                    r'^(?:premium|standard|basic|gold|silver|bronze|platinum|vip|regular)$',
                    r'^tier[\s_-]?\d$',
                    r'^[A-D]$'  # Simple letter-based segments
                ]
            },
            'customer_acquisition': {
                'name_patterns': [
                    r'\b(?:customer|client|lead|prospect)[\s_-]*(?:acquisition|source|origin|channel)\b',
                    r'\b(?:acquisition|conversion|signup|registration|join|onboarding)[\s_-]*(?:source|channel|medium|campaign)\b',
                    r'\b(?:lead|referral|traffic)[\s_-]*source\b'
                ],
                'value_patterns': [
                    r'^(?:web|email|social|referral|organic|paid|direct|offline|store|call|event)$',
                    r'^(?:facebook|google|twitter|linkedin|instagram|tiktok|youtube)$'
                ]
            },
            'customer_ltv': {
                'name_patterns': [
                    r'\b(?:customer|client|buyer|account)[\s_-]*(?:ltv|lifetime[\s_-]*value|clv|value)\b',
                    r'\b(?:predicted|forecasted|estimated|projected)[\s_-]*(?:ltv|lifetime[\s_-]*value|clv)\b',
                    r'\b(?:ltv|clv|cltv)[\s_-]*(?:value|amount|score)?\b'
                ]
            },
            'customer_contact': {
                'name_patterns': [
                    r'\b(?:email|e-?mail|mail)[\s_-]*(?:address)?\b',
                    r'\b(?:phone|telephone|mobile|cell)[\s_-]*(?:number|no)?\b',
                    r'\bcontact[\s_-]*(?:details|info)\b'
                ],
                'value_patterns': {
                    'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                    'phone': r'^\+?[\d\s-\(\).]{7,}$'
                }
            },
            'customer_address': {
                'name_patterns': [
                    r'\b(?:billing|shipping|mailing|delivery|home|work|office)[\s_-]*address\b',
                    r'\b(?:street|avenue|road|boulevard|lane|plaza|building|apt|suite)[\s_-]*(?:address|number|no)?\b',
                    r'\b(?:address|addr)[\s_-]*(?:line[\s_-]*[1-3]|1|2|3)\b'
                ]
            },
            'customer_status': {
                'name_patterns': [
                    r'\b(?:customer|client|account)[\s_-]*(?:status|state|condition|standing)\b',
                    r'\b(?:active|inactive|suspended|pending|canceled|terminated|churned)[\s_-]*(?:status|flag|indicator)?\b',
                    r'\b(?:status|state)[\s_-]*(?:code|value|indicator)?\b'
                ],
                'value_patterns': [
                    r'^(?:active|inactive|pending|suspended|closed|terminated|on[\s_-]*hold|new|verified)$',
                    r'^(?:0|1|Y|N|A|I|P)$'  # Common status codes
                ]
            },
            'customer_age': {
                'name_patterns': [
                    r'\b(?:customer|client|user|member|account)[\s_-]*(?:age|years|yrs)\b',
                    r'\b(?:age|years|yrs)[\s_-]*(?:old)?\b',
                    r'\b(?:birth|dob|birth[\s_-]*date)[\s_-]*(?:date|day|year)?\b'
                ],
                'value_patterns': [
                    r'^(?:1[89]|[2-9][0-9])$',  # Ages 18-99
                    r'^\d{1,2}/\d{1,2}/\d{4}$',  # MM/DD/YYYY
                    r'^\d{4}-\d{2}-\d{2}$'  # YYYY-MM-DD
                ]
            },
            'customer_gender': {
                'name_patterns': [
                    r'\b(?:gender|sex|male|female)\b',
                    r'\b(?:customer|client|user|member|account)[\s_-]*(?:gender|sex)\b'
                ],
                'value_patterns': [
                    r'^(?:M|F|Male|Female|man|woman|non[\s_-]*binary|other)$',
                    r'^(?:m|f|male|female)$'
                ]
            },
            'customer_purchase_frequency': {
                'name_patterns': [
                    r'\b(?:customer|client|buyer|account)[\s_-]*(?:purchase|order|transaction)[\s_-]*(?:frequency|rate|count|number)\b',
                    r'\b(?:purchase|order|buy|transaction)[\s_-]*(?:frequency|rate|cadence|pattern|cycle)\b',
                    r'\b(?:frequency|times|count)[\s_-]*(?:of|per)[\s_-]*(?:purchase|order|transaction)\b'
                ]
            },
            'customer_churn_risk': {
                'name_patterns': [
                    r'\b(?:customer|client|account|churn|attrition)[\s_-]*(?:risk|score|probability|likelihood|potential)\b',
                    r'\b(?:churn|attrition|loss|cancellation|defection)[\s_-]*(?:risk|score|rate|prediction|probability)\b',
                    r'\b(?:risk|likelihood|probability)[\s_-]*(?:of|to)[\s_-]*(?:churn|cancel|leave|attrition)\b'
                ],
                'value_patterns': [
                    r'^(?:high|medium|low|H|M|L|critical|warning|safe)$',
                    r'^(?:0|[0-9]\.[0-9]+|[0-9]{1,2}|100)%?$'  # 0-100% or 0.0-1.0
                ]
            }
        }
    
    def _initialize_time_patterns(self):
        """Initialize time-related patterns"""
        return {
            'time_of_day': {
                'name_patterns': [
                    r'\b(?:time|hour|minute|second)[\s_-]*(?:of[\s_-]*day)?\b',
                    r'\b(?:timestamp|datetime|datetimeoffset)\b',
                    r'\b(?:clock|wall)[\s_-]*time\b'
                ],
                'value_patterns': [
                    r'^\d{1,2}:\d{2}(:\d{2})?([aApP][mM])?$',  # 3:30PM, 15:45, 9:20:15
                    r'^\d{1,2}[aApP][mM]$'  # 3PM, 11am
                ]
            },
            'duration': {
                'name_patterns': [
                    r'\b(?:duration|elapsed|length|span|interval|period)[\s_-]*(?:time|minutes|seconds|hours)?\b',
                    r'\b(?:time[\s_-]*spent|time[\s_-]*elapsed|time[\s_-]*taken)\b',
                    r'\b(?:session|call|visit|stay|usage)[\s_-]*(?:length|duration|time)\b'
                ],
                'value_patterns': [
                    r'^\d+:?\d*:?\d*$',  # 30, 1:30, 2:45:30
                    r'^\d+[\s_-]*(?:ms|sec|min|hr|day|wk|mo|yr)$'  # 30sec, 45min, 2hr
                ]
            },
            'delivery_time': {
                'name_patterns': [
                    r'\b(?:delivery|shipping|arrival|fulfillment|transit)[\s_-]*(?:time|date|timeframe|window|period|schedule)\b',
                    r'\b(?:estimated|actual|scheduled|promised|target)[\s_-]*(?:delivery|arrival|ship)[\s_-]*(?:date|time)\b',
                    r'\b(?:eta|etd|ata|atd)[\s_-]*(?:date|time)?\b'  # Estimated/Actual Time of Arrival/Departure
                ]
            },
            'lead_time': {
                'name_patterns': [
                    r'\b(?:lead|lag|processing|handling|turnaround|response)[\s_-]*time\b',
                    r'\b(?:time[\s_-]*to[\s_-]*(?:process|fulfill|complete|respond|ship|deliver))\b',
                    r'\b(?:sla|service[\s_-]*level[\s_-]*agreement)[\s_-]*(?:time|hours|days)?\b'
                ]
            },
            'time_zone': {
                'name_patterns': [
                    r'\b(?:time[\s_-]*zone|tz|timezone|time[\s_-]*offset|utc[\s_-]*offset)\b',
                    r'\b(?:local|server|system|user)[\s_-]*(?:time[\s_-]*zone|tz)\b'
                ],
                'value_patterns': [
                    r'^(?:UTC|GMT|EST|CST|MST|PST|EDT|CDT|MDT|PDT)[\+\-]?\d*$',
                    r'^[\+\-]\d{1,2}(?::\d{2})?$'  # +8, -5:30
                ]
            },
            'frequency': {
                'name_patterns': [
                    r'\b(?:frequency|interval|periodicity|recurrence|cycle|cadence)\b',
                    r'\b(?:daily|weekly|monthly|quarterly|yearly|annual)[\s_-]*(?:frequency|occurrence|schedule)?\b',
                    r'\b(?:times[\s_-]*per[\s_-]*(?:day|week|month|year|quarter))\b'
                ],
                'value_patterns': [
                    r'^(?:daily|weekly|biweekly|monthly|quarterly|annually|hourly|minutely)$',
                    r'^(?:every|once)[\s_-]+(?:day|week|month|year|hour|minute|second)$',
                    r'^\d+[\s_-]*(?:ms|sec|min|hr|day|wk|mo|yr)$'
                ]
            },
            'datetime_components': {
                'name_patterns': [
                    r'\b(?:year|yr|yyyy|fiscal[\s_-]*year|calendar[\s_-]*year)\b',
                    r'\b(?:month|mon|mm|quarter|qtr|season)\b',
                    r'\b(?:day|dy|dd|weekday|dow|dom)\b',
                    r'\b(?:hour|hr|minute|min|second|sec)\b'
                ],
                'value_patterns': {
                    'year': r'^(?:19|20)\d{2}$',  # 1900-2099
                    'month': r'^(?:0?[1-9]|1[0-2])$|^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$',
                    'day': r'^(?:0?[1-9]|[12][0-9]|3[01])$',
                    'hour': r'^(?:[01]?[0-9]|2[0-3])$',
                    'minute': r'^(?:[0-5]?[0-9])$',
                }
            },
            'week': {
                'name_patterns': [
                    r'\b(?:week|wk|week[\s_-]*number|week[\s_-]*no|weeknum)\b',
                    r'\b(?:iso|calendar|fiscal)[\s_-]*week\b',
                    r'\b(?:woy|wow|ww)[\s_-]*(?:number|no)?\b'  # Week of year, Week over week, Work week
                ],
                'value_patterns': [
                    r'^(?:[0-9]|[1-4][0-9]|5[0-3])$',  # 0-53
                    r'^W(?:[0-9]|[1-4][0-9]|5[0-3])$'  # W1-W53
                ]
            }
        }
    
    def _initialize_location_patterns(self):
        """Initialize geographic/location patterns"""
        return {
            'address': {
                'name_patterns': [
                    r'\b(?:address|addr|location|street|ave|blvd|road|rd)[\s_-]*(?:line)?[\s_-]*(?:1|2|3|one|two|three)?\b',
                    r'\b(?:billing|shipping|mailing|delivery|home|work|business)[\s_-]*address\b',
                    r'\b(?:street|building|location)[\s_-]*(?:address|number|name)\b'
                ]
            },
            'city': {
                'name_patterns': [
                    r'\b(?:city|town|municipality|borough|village|suburb|settlement|urban[\s_-]*area)\b',
                    r'\b(?:billing|shipping|mailing|delivery|home|work|business)[\s_-]*city\b',
                    r'\b(?:city|town)[\s_-]*(?:name|code)?\b'
                ]
            },
            'state_province': {
                'name_patterns': [
                    r'\b(?:state|province|county|region|district|territory|prefecture)\b',
                    r'\b(?:billing|shipping|mailing|delivery|home|work|business)[\s_-]*(?:state|province)\b',
                    r'\b(?:state|province)[\s_-]*(?:name|code|abbr|abbreviation)?\b'
                ],
                'value_patterns': [
                    r'^[A-Z]{2}$',  # US state codes: NY, CA, etc.
                    r'^(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)$'  # Specific US states
                ]
            },
            'postal_code': {
                'name_patterns': [
                    r'\b(?:zip|postal|post)[\s_-]*(?:code)?\b',
                    r'\b(?:billing|shipping|mailing|delivery|home|work|business)[\s_-]*(?:zip|postal|post)[\s_-]*(?:code)?\b'
                ],
                'value_patterns': [
                    r'^\d{5}(?:-\d{4})?$',  # US ZIP: 12345 or 12345-6789
                    r'^[A-Z]\d[A-Z][\s-]?\d[A-Z]\d$'  # Canadian postal code: A1A 1A1
                ]
            },
            'country': {
                'name_patterns': [
                    r'\b(?:country|nation|land|territory|commonwealth|republic|kingdom)\b',
                    r'\b(?:billing|shipping|mailing|delivery|home|work|business)[\s_-]*country\b',
                    r'\b(?:country|nation)[\s_-]*(?:name|code|iso)?\b'
                ],
                'value_patterns': [
                    r'^(?:US|USA|CA|CAN|UK|GB|AU|DE|FR|JP|CN|IN|BR)$',  # Common ISO codes
                    r'^[A-Z]{2}$',  # ISO 2-letter codes
                    r'^[A-Z]{3}$'   # ISO 3-letter codes
                ]
            },
            'region': {
                'name_patterns': [
                    r'\b(?:region|area|zone|district|sector|territory|jurisdiction)\b',
                    r'\b(?:sales|service|delivery|market)[\s_-]*(?:region|area|zone|territory)\b',
                    r'\b(?:region|zone)[\s_-]*(?:name|code|id)?\b'
                ]
            },
            'geo_coordinates': {
                'name_patterns': [
                    r'\b(?:latitude|lat|longitude|long|lon|lng|coords|coordinates|geo|gps|position)\b',
                    r'\b(?:lat|latitude)[\s_-]*(?:value|coordinate|position|degrees)?\b',
                    r'\b(?:lon|long|longitude)[\s_-]*(?:value|coordinate|position|degrees)?\b'
                ],
                'value_patterns': {
                    'latitude': r'^-?(?:90|[1-8]?[0-9](?:\.\d+)?)$',
                    'longitude': r'^-?(?:180|1[0-7][0-9]|[1-9]?[0-9](?:\.\d+)?)$'
                }
            },
            'store_location': {
                'name_patterns': [
                    r'\b(?:store|shop|outlet|branch|location|site|dealer|franchise)[\s_-]*(?:id|code|number|location|address)?\b',
                    r'\b(?:retail|warehouse|distribution|pickup|collection)[\s_-]*(?:location|site|center|facility)\b'
                ],
                'value_patterns': [
                    r'^[A-Z]{1,2}[-_]?\d{2,4}$',  # Store codes like NY-123, TX12
                    r'^ST\d{3,}$',  # ST123, ST4567
                    r'^\d{3,5}$'    # Simple numeric store IDs
                ]
            }
        }
    
    def _initialize_marketing_patterns(self):
        """Initialize marketing and campaign patterns"""
        return {
            'campaign': {
                'name_patterns': [
                    r'\b(?:campaign|promotion|promo|initiative|drive|marketing)[\s_-]*(?:id|code|name|number|no|identifier)?\b',
                    r'\b(?:advertisement|ad|advert)[\s_-]*(?:campaign|id|code|name|number)?\b',
                    r'\b(?:camp|cmpgn)[\s_-]*(?:id|code|num|no)?\b'
                ]
            },
            'channel': {
                'name_patterns': [
                    r'\b(?:channel|medium|platform|source|touchpoint|outlet)\b',
                    r'\b(?:marketing|sales|distribution|communication|acquisition)[\s_-]*(?:channel|source|medium)\b',
                    r'\b(?:utm|tracking)[\s_-]*(?:source|medium|channel)\b'
                ],
                'value_patterns': [
                    r'^(?:web|email|social|print|tv|radio|direct|mail|store|online|offline|mobile|app|referral)$',
                    r'^(?:facebook|twitter|instagram|linkedin|youtube|google|pinterest|tiktok)$'
                ]
            },
            'promotion': {
                'name_patterns': [
                    r'\b(?:promotion|promo|offer|deal|discount|special)[\s_-]*(?:id|code|name|type)?\b',
                    r'\b(?:coupon|voucher|rebate)[\s_-]*(?:code|id|number|amount|value|percent)?\b',
                    r'\b(?:promo|discount)[\s_-]*(?:rate|amount|value|percentage|level|tier)?\b'
                ]
            },
            'segment': {
                'name_patterns': [
                    r'\b(?:segment|segmentation|audience|cohort|group|cluster|category)\b',
                    r'\b(?:market|customer|user|buyer|client|visitor)[\s_-]*(?:segment|group|cluster|category|cohort)\b',
                    r'\b(?:demographic|psychographic|behavioral)[\s_-]*(?:segment|group|cluster|category)?\b'
                ]
            },
            'conversion': {
                'name_patterns': [
                    r'\b(?:conversion|click|action|engagement|interaction|event)[\s_-]*(?:rate|percentage|count|number|value)?\b',
                    r'\b(?:bounce|exit|abandonment)[\s_-]*rate\b',
                    r'\b(?:ctr|cvr|cpa|cpc|cpm)[\s_-]*(?:value|rate|amount)?\b'  # Click-through rate, Conversion rate, etc.
                ]
            },
            'attribution': {
                'name_patterns': [
                    r'\b(?:attribution|contribution|credit|source|origin|influence)\b',
                    r'\b(?:first|last|multi|linear|time|position|touch)[\s_-]*(?:touch|click|interaction|attribution|credit)\b',
                    r'\b(?:attribution|contribution)[\s_-]*(?:model|method|approach|algorithm|rule|logic)\b'
                ],
                'value_patterns': [
                    r'^(?:first|last|linear|time|position|multi|data)[\s_-]*(?:touch|click|interaction|attribution)$',
                    r'^(?:direct|organic|referral|social|email|paid|affiliate|partner)$'
                ]
            },
            'marketing_cost': {
                'name_patterns': [
                    r'\b(?:marketing|advertising|promotion|campaign)[\s_-]*(?:cost|expense|spend|budget|investment)\b',
                    r'\b(?:ad|campaign|promotion)[\s_-]*(?:spend|cost|expense|budget)\b',
                    r'\b(?:cac|cpa|cpc|cpm|cpp)[\s_-]*(?:cost|value|amount|rate)?\b'  # Cost per acquisition, Cost per click, etc.
                ]
            },
            'marketing_roi': {
                'name_patterns': [
                    r'\b(?:marketing|advertising|promotion|campaign)[\s_-]*(?:roi|return|roas|performance|efficiency)\b',
                    r'\b(?:roi|roas|romi|return)[\s_-]*(?:on|of)[\s_-]*(?:marketing|advertising|ad|campaign|promotion)[\s_-]*(?:spend|investment|expense)?\b',
                    r'\b(?:campaign|ad|promotion)[\s_-]*(?:roi|return|performance|results|success|effectiveness)\b'
                ],
                'value_patterns': [
                    r'^-?\d+(?:\.\d+)?%?$',  # Numeric ROI (possibly with percentage sign)
                    r'^-?\d+(?:\.\d+)?[xX]$'  # ROI in format like "3.5x"
                ]
            }
        }
    
    def _initialize_operational_patterns(self):
        """Initialize operational/fulfillment patterns"""
        return {
            'order_status': {
                'name_patterns': [
                    r'\b(?:order|shipment|delivery|fulfillment|transaction)[\s_-]*(?:status|state|condition|stage|phase)\b',
                    r'\b(?:current|latest|updated|tracking)[\s_-]*(?:status|state|condition)\b',
                    r'\b(?:status|state)[\s_-]*(?:code|value|indicator|flag)?\b'
                ],
                'value_patterns': [
                    r'^(?:new|pending|processing|shipped|delivered|completed|cancelled|returned|on[-\s_]*hold|back[-\s_]*ordered)$',
                    r'^(?:N|P|S|D|C|X|R|H|B)$'  # Status codes
                ]
            },
            'shipping_method': {
                'name_patterns': [
                    r'\b(?:shipping|delivery|transport|carrier|shipment|fulfillment)[\s_-]*(?:method|type|mode|service|option|provider|carrier|company)?\b',
                    r'\b(?:express|standard|overnight|priority|ground|air|freight)[\s_-]*(?:shipping|delivery|service)?\b',
                    r'\b(?:ship|delivery)[\s_-]*(?:via|by|through|method|type|mode)\b'
                ],
                'value_patterns': [
                    r'^(?:standard|express|overnight|priority|ground|air|2[-\s_]*day|next[-\s_]*day)$',
                    r'^(?:fedex|ups|usps|dhl|amazon|royal[-\s_]*mail|canada[-\s_]*post)$'
                ]
            },
            'tracking_id': {
                'name_patterns': [
                    r'\b(?:tracking|shipment|package|parcel|delivery)[\s_-]*(?:id|number|code|identifier|reference|no)\b',
                    r'\b(?:track|trace)[\s_-]*(?:id|no|number|code)\b',
                    r'\b(?:waybill|airway[-\s_]*bill|bill[-\s_]*of[-\s_]*lading)[\s_-]*(?:number|no|id|code)?\b'
                ],
                'value_patterns': [
                    r'^[0-9]{8,15}$',  # Basic numeric tracking
                    r'^[A-Z]{2}[0-9]{9}[A-Z]{2}$',  # USPS format
                    r'^1Z[A-Z0-9]{16}$'  # UPS format
                ]
            },
            'fulfillment_center': {
                'name_patterns': [
                    r'\b(?:fulfillment|distribution|warehouse|storage|inventory|logistics)[\s_-]*(?:center|facility|location|building|site|hub|depot)\b',
                    r'\b(?:fc|dc|wh)[\s_-]*(?:id|code|number|name|location)?\b',
                    r'\b(?:shipping|fulfillment|shipping)[\s_-]*(?:from|origin|source|location)\b'
                ]
            },
            'inventory_status': {
                'name_patterns': [
                    r'\b(?:inventory|stock|supply|availability|quantity)[\s_-]*(?:status|level|state|condition|position)\b',
                    r'\b(?:in[-\s_]*stock|out[-\s_]*of[-\s_]*stock|available|unavailable|backorder)[\s_-]*(?:status|flag|indicator)?\b',
                    r'\b(?:stock|inventory)[\s_-]*(?:on[-\s_]*hand|available|reserved|allocated|committed)\b'
                ],
                'value_patterns': [
                    r'^(?:in[-\s_]*stock|out[-\s_]*of[-\s_]*stock|low[-\s_]*stock|available|unavailable|backorder)$',
                    r'^(?:1|0|Y|N|A|U|L|B)$'  # Status codes (1/0, Yes/No, etc.)
                ]
            },
            'batch_number': {
                'name_patterns': [
                    r'\b(?:batch|lot|production|manufacturing)[\s_-]*(?:number|no|id|code|identifier)\b',
                    r'\b(?:batch|lot)[\s_-]*(?:qty|quantity|volume|amount|count)?\b',
                    r'\b(?:production|process|manufacturing)[\s_-]*(?:run|sequence|series)[\s_-]*(?:id|no|number)?\b'
                ],
                'value_patterns': [
                    r'^[A-Z]{1,3}[0-9]{4,8}$',  # BA12345
                    r'^[0-9]{4,8}[A-Z]{1,3}$',  # 12345BA
                    r'^(?:B|L|P)-[0-9]{4,8}$'  # B-12345
                ]
            },
            'order_id': {
                'name_patterns': [
                    r'\b(?:order|purchase|transaction|invoice|sales|checkout)[\s_-]*(?:id|number|no|code|identifier|reference)\b',
                    r'\b(?:ord|po|pur|inv)[\s_-]*(?:id|no|number|code|#)\b',
                    r'\b(?:confirmation|receipt)[\s_-]*(?:number|no|id|code)\b'
                ],
                'value_patterns': [
                    r'^ORD-?\d{5,10}$',  # ORD12345, ORD-12345
                    r'^PO-?\d{5,10}$',   # PO12345, PO-12345
                    r'^\d{5,10}$'        # Simple numeric order IDs
                ]
            },
            'order_line': {
                'name_patterns': [
                    r'\b(?:order|line|item)[\s_-]*(?:line|item|position|number|sequence|sequence[\s_-]*number)\b',
                    r'\b(?:line|item)[\s_-]*(?:in|on|of)[\s_-]*(?:order|invoice|receipt)\b',
                    r'\b(?:item|position|line)[\s_-]*(?:#|no|number)\b'
                ],
                'value_patterns': [
                    r'^\d{1,3}$',  # Simple line numbers (1, 2, 3, etc.)
                    r'^\d{1,3}\.\d{1,3}$'  # Hierarchical line numbers (1.1, 1.2, etc.)
                ]
            }
        }
    
    def _initialize_temporal_patterns(self):
        """Initialize temporal patterns (expanded date/time detection)"""
        return {
            'year': {
                'name_patterns': [
                    r'\b(?:year|yr|yyyy|fiscal[\s_-]*year|fy|annual|calendar[\s_-]*year)\b',
                    r'\b(?:year|yr)[\s_-]*(?:of|in)[\s_-]*(?:order|purchase|transaction|sale|report)\b'
                ],
                'value_patterns': [
                    r'^(?:19|20)\d{2}$',  # 1900-2099
                    r'^\d{2}$',  # 2-digit years (22, 23, etc.)
                    r'^FY\d{2}(?:\d{2})?$'  # Fiscal year formats (FY22, FY2022)
                ]
            },
            'month': {
                'name_patterns': [
                    r'\b(?:month|mon|mm|mo|mnth)\b',
                    r'\b(?:month|mon)[\s_-]*(?:of|in)[\s_-]*(?:order|purchase|transaction|sale|year|report)\b',
                    r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(?:uary|ruary|ch|il|ust|tember|ober|ember)?\b'
                ],
                'value_patterns': [
                    r'^(?:0?[1-9]|1[0-2])$',  # 1-12 (or 01-12)
                    r'^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$',  # Three-letter abbreviations
                    r'^(?:January|February|March|April|May|June|July|August|September|October|November|December)$'  # Full names
                ]
            },
            'quarter': {
                'name_patterns': [
                    r'\b(?:quarter|qtr|q[1-4]|fiscal[\s_-]*quarter)\b',
                    r'\b(?:quarter|qtr)[\s_-]*(?:[1-4]|one|two|three|four)\b',
                    r'\b(?:q|quarter)[\s_-]*(?:of|in)[\s_-]*(?:year|fiscal|fy)\b'
                ],
                'value_patterns': [
                    r'^[Qq][1-4]$',  # Q1, Q2, Q3, Q4 (or q1, q2, q3, q4)
                    r'^[1-4]$',  # Simple 1, 2, 3, 4
                    r'^(?:first|second|third|fourth)[\s_-]*(?:quarter|qtr)$'  # Text formats
                ]
            },
            'day_of_week': {
                'name_patterns': [
                    r'\b(?:day[\s_-]*of[\s_-]*week|weekday|dow|day[\s_-]*name)\b',
                    r'\b(?:mon|tue|wed|thu|fri|sat|sun)(?:day|\.)?[\s_-]*(?:indicator|flag)?\b'
                ],
                'value_patterns': [
                    r'^(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)$',
                    r'^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)$',
                    r'^[0-6]$'  # 0-6 representation (often 0=Sunday or 0=Monday)
                ]
            },
            'day_of_month': {
                'name_patterns': [
                    r'\b(?:day|dy|dd|dom|day[\s_-]*of[\s_-]*month)\b',
                    r'\b(?:day)[\s_-]*(?:in|of)[\s_-]*(?:month|order|transaction|report)\b'
                ],
                'value_patterns': [
                    r'^(?:[1-9]|[12][0-9]|3[01])$',  # 1-31
                    r'^(?:0[1-9]|[12][0-9]|3[01])$'  # 01-31
                ]
            },
            'day_of_year': {
                'name_patterns': [
                    r'\b(?:day[\s_-]*of[\s_-]*year|doy|julian[\s_-]*day|ordinal[\s_-]*day)\b',
                    r'\b(?:yearly|annual)[\s_-]*(?:day|date)[\s_-]*(?:number|no)?\b'
                ],
                'value_patterns': [
                    r'^(?:[1-9]|[1-9][0-9]|[12][0-9]{2}|3[0-5][0-9]|36[0-6])$',  # 1-366
                    r'^(?:00[1-9]|0[1-9][0-9]|[12][0-9]{2}|3[0-5][0-9]|36[0-6])$'  # 001-366
                ]
            },
            'hour': {
                'name_patterns': [
                    r'\b(?:hour|hr|hrs|hh)\b',
                    r'\b(?:hour)[\s_-]*(?:of|in)[\s_-]*(?:day|order|transaction|report)\b'
                ],
                'value_patterns': [
                    r'^(?:[0-9]|1[0-9]|2[0-3])$',  # 0-23
                    r'^(?:0[0-9]|1[0-9]|2[0-3])$',  # 00-23
                    r'^(?:[1-9]|1[0-2])[\s_-]*(?:am|pm|AM|PM)$'  # 12-hour format
                ]
            },
            'minute': {
                'name_patterns': [
                    r'\b(?:minute|min|mm)\b',
                    r'\b(?:minute)[\s_-]*(?:of|in)[\s_-]*(?:hour|time|timestamp)\b'
                ],
                'value_patterns': [
                    r'^(?:[0-9]|[1-5][0-9])$',  # 0-59
                    r'^(?:0[0-9]|[1-5][0-9])$'  # 00-59
                ]
            },
            'second': {
                'name_patterns': [
                    r'\b(?:second|sec|ss)\b',
                    r'\b(?:second)[\s_-]*(?:of|in)[\s_-]*(?:minute|time|timestamp)\b'
                ],
                'value_patterns': [
                    r'^(?:[0-9]|[1-5][0-9])$',  # 0-59
                    r'^(?:0[0-9]|[1-5][0-9])$'  # 00-59
                ]
            },
            'period': {
                'name_patterns': [
                    r'\b(?:period|time[\s_-]*period|date[\s_-]*range|time[\s_-]*range|time[\s_-]*frame|window)\b',
                    r'\b(?:reporting|analysis|accounting|sales|fiscal)[\s_-]*(?:period|timeframe|time[\s_-]*frame)\b',
                    r'\b(?:from|start|begin|to|end|through)[\s_-]*(?:date|time|period)\b'
                ]
            },
            'season': {
                'name_patterns': [
                    r'\b(?:season|seasonal|quarter|period)\b',
                    r'\b(?:spring|summer|fall|autumn|winter)[\s_-]*(?:season|period|quarter)?\b',
                    r'\b(?:holiday|shopping|peak|off[\s_-]*peak)[\s_-]*(?:season|period|time)\b'
                ],
                'value_patterns': [
                    r'^(?:spring|summer|fall|autumn|winter)$',
                    r'^(?:q[1-4]|quarter[\s_-]*[1-4])$',
                    r'^(?:holiday|peak|off[\s_-]*peak|high|low)[\s_-]*(?:season)?$'
                ]
            }
        }
    
    def _initialize_financial_patterns(self):
        """Initialize financial patterns (expanded sales/revenue detection)"""
        return {
            'revenue': {
                'name_patterns': [
                    r'\b(?:revenue|sales|income|proceeds|turnover|earnings)\b',
                    r'\b(?:gross|net|total|monthly|quarterly|annual|daily)[\s_-]*(?:revenue|sales|income|proceeds)\b',
                    r'\b(?:revenue|sales)[\s_-]*(?:amount|figure|number|total|value|volume)\b'
                ]
            },
            'price': {
                'name_patterns': [
                    r'\b(?:price|cost|rate|charge|fee)\b',
                    r'\b(?:unit|per[\s_-]*item|single|individual)[\s_-]*(?:price|cost|rate|charge|fee)\b',
                    r'\b(?:price|cost)[\s_-]*(?:per|each|unit|item)\b'
                ]
            },
            'discount': {
                'name_patterns': [
                    r'\b(?:discount|reduction|markdown|savings|sale|promotion)\b',
                    r'\b(?:price|cost)[\s_-]*(?:discount|reduction|markdown|adjustment|deduction)\b',
                    r'\b(?:discount|promo|coupon|voucher)[\s_-]*(?:amount|value|percentage|rate|code)\b'
                ],
                'value_patterns': [
                    r'^-?\d+(?:\.\d+)?%?$',  # Numeric discount (may include % sign)
                    r'^-?\d+(?:\.\d+)?[\s_-]*(?:off|discount)$'  # Format like "25 off"
                ]
            },
            'tax': {
                'name_patterns': [
                    r'\b(?:tax|vat|gst|sales[\s_-]*tax|use[\s_-]*tax)\b',
                    r'\b(?:tax|vat|gst)[\s_-]*(?:amount|value|rate|percentage|total)\b',
                    r'\b(?:federal|state|local|city|county|provincial)[\s_-]*tax\b'
                ],
                'value_patterns': [
                    r'^\d+(?:\.\d+)?%?$'  # Tax rate (may include % sign)
                ]
            },
            'shipping': {
                'name_patterns': [
                    r'\b(?:shipping|freight|delivery|transport|postage|handling)\b',
                    r'\b(?:shipping|freight|delivery|transport)[\s_-]*(?:cost|charge|fee|expense|amount)\b',
                    r'\b(?:shipping|delivery)[\s_-]*(?:rate|price|cost|charge)\b'
                ]
            },
            'total': {
                'name_patterns': [
                    r'\b(?:total|sum|grand[\s_-]*total|final|overall)\b',
                    r'\b(?:total|final|overall)[\s_-]*(?:amount|cost|price|value|charge|fee)\b',
                    r'\b(?:order|invoice|purchase|transaction)[\s_-]*(?:total|amount|value|sum)\b'
                ]
            },
            'refund': {
                'name_patterns': [
                    r'\b(?:refund|return|reimbursement|chargeback|credit|money[\s_-]*back)\b',
                    r'\b(?:refund|return|reimbursement)[\s_-]*(?:amount|value|total|sum)\b',
                    r'\b(?:refunded|returned|credited|reimbursed)[\s_-]*(?:amount|value|total|sum)\b'
                ]
            },
            'cost': {
                'name_patterns': [
                    r'\b(?:cost|expense|expenditure|outlay|spend|spending)\b',
                    r'\b(?:cost|purchase|wholesale|acquisition)[\s_-]*(?:price|amount|value)\b',
                    r'\b(?:unit|per[\s_-]*item|manufacturing|production)[\s_-]*cost\b'
                ]
            },
            'margin': {
                'name_patterns': [
                    r'\b(?:margin|markup|profit[\s_-]*margin|contribution[\s_-]*margin)\b',
                    r'\b(?:gross|net|operating|profit)[\s_-]*margin\b',
                    r'\b(?:margin|markup)[\s_-]*(?:rate|percentage|amount|value)\b'
                ],
                'value_patterns': [
                    r'^\d+(?:\.\d+)?%?$'  # Margin rate (may include % sign)
                ]
            },
            'profit': {
                'name_patterns': [
                    r'\b(?:profit|earning|gain|surplus|advantage|yield|return)\b',
                    r'\b(?:gross|net|operating|pre[\s_-]*tax|after[\s_-]*tax)[\s_-]*(?:profit|earnings|income)\b',
                    r'\b(?:profit|earning)[\s_-]*(?:amount|value|total|sum)\b'
                ]
            },
            'currency': {
                'name_patterns': [
                    r'\b(?:currency|money|denomination|tender|legal[\s_-]*tender)\b',
                    r'\b(?:currency|monetary)[\s_-]*(?:code|unit|type|sign|symbol)\b',
                    r'\b(?:payment|transaction)[\s_-]*currency\b'
                ],
                'value_patterns': [
                    r'^(?:USD|EUR|GBP|JPY|CNY|AUD|CAD|CHF|INR|BRL)$',  # Common ISO codes
                    r'^[A-Z]{3}$',  # Any 3-letter code
                    r'^(?:\$|€|£|¥|₹|₽|₩|₿)$'  # Currency symbols
                ]
            }
        }
    
    def _initialize_product_patterns(self):
        """Initialize product patterns (expanded product detection)"""
        return {
            'product_id': {
                'name_patterns': [
                    r'\b(?:product|item|article|good|sku|upc|ean|isbn)[\s_-]*(?:id|code|number|no|identifier|key)\b',
                    r'\b(?:prod|prd|itm|art)[\s_-]*(?:id|code|num|no)\b',
                    r'\b(?:bar|qr)[\s_-]*code\b'
                ],
                'value_patterns': [
                    r'^[A-Z]{1,3}[-_]?\d{3,}$',  # ABC123, A-123, etc.
                    r'^\d{3,}[-_]?[A-Z]{1,3}$',  # 123ABC, 123-A, etc.
                    r'^(?:\d{8}|\d{12}|\d{13}|\d{14})$'  # UPC/EAN/ISBN formats
                ]
            },
            'product_name': {
                'name_patterns': [
                    r'\b(?:product|item|article|good|merchandise)[\s_-]*(?:name|title|label|description|desc)\b',
                    r'\b(?:prod|prd|itm|art)[\s_-]*(?:name|desc)\b',
                    r'\bdescription\b'
                ]
            },
            'product_category': {
                'name_patterns': [
                    r'\b(?:product|item)[\s_-]*(?:category|cat|type|group|class|classification|family)\b',
                    r'\b(?:category|cat|type|group|class|classification|family)[\s_-]*(?:of[\s_-]*(?:product|item))?\b',
                    r'\b(?:main|primary|secondary|sub)[\s_-]*(?:category|cat)\b'
                ],
                'value_patterns': [
                    r'^(?:electronics|clothing|food|beverage|furniture|home|garden|tools|automotive|beauty|health|toys|sports|books|music)$'
                ]
            },
            'product_brand': {
                'name_patterns': [
                    r'\b(?:brand|make|manufacturer|vendor|supplier|producer|creator)\b',
                    r'\b(?:product|item)[\s_-]*(?:brand|make|manufacturer|vendor)\b',
                    r'\b(?:brand|make|manufacturer)[\s_-]*(?:name|id|code)\b'
                ]
            },
          
            'product': {
                'name_patterns': [
                    r'\b(?:product|item|sku|merchandise|good|article|material)\b',
                    r'\b(?:product|item|article)[\s_-]*(?:id|code|number|no|identifier|key)\b'
                ],
                'id_patterns': [
                    r'^[A-Z]{1,3}[-_]?\d{3,}$',            # ABC123, A-123, etc.
                    r'^\d{3,}[-_]?[A-Z]{1,3}$',            # 123ABC, 123-A, etc.
                    r'^[A-Z0-9]{2,3}[-_]?\d{4,}$',         # AB1234, XYZ9876, etc.
                    r'^(?:[A-Z0-9]{2,5}[-_]){1,2}[A-Z0-9]{2,5}$'  # AB-12-XY, 123-ABC-45, etc.
                ]
            },
            'sales': {
                'name_patterns': [
                    r'\b(?:sales|revenue|amount|value|total|sum|selling)\b',
                    r'\b(?:total|gross)[\s_-]*(?:sales|revenue|income|amount|value|price)\b',
                    r'\b(?:extended|line)[\s_-]*(?:amount|price|total|value)\b'
                ]
            },
            'quantity': {
                'name_patterns': [
                    r'\b(?:quantity|qty|count|units|volume|number|pieces)\b',
                    r'\b(?:number|count)[\s_-]*(?:of|items|units|pieces)\b',
                    r'\b(?:item|unit|piece)[\s_-]*count\b'
                ],
                'value_patterns': [
                    r'^\d+$',  # Integer values
                ]
            },
            'price': {
                'name_patterns': [
                    r'\b(?:price|cost|rate|fee|charge)\b',
                    r'\b(?:unit|per[\s_-]*item)[\s_-]*(?:price|cost|value|amount)\b',
                    r'\b(?:price|cost)[\s_-]*(?:per|each|unit|item)\b'
                ]
            },
            'boolean': {
                'name_patterns': [
                    r'\b(?:is|has|should|can|will|flag|indicator|status|state)\b',
                    r'\b(?:is[\s_-]*(?:active|enabled|valid|canceled|approved|completed))\b',
                    r'\b(?:has[\s_-]*(?:approved|verified|validated|confirmed))\b'
                ],
                'value_patterns': [
                    r'^(?:0|1)$',
                    r'^(?:true|false)$',
                    r'^(?:yes|no)$',
                    r'^(?:y|n)$',
                    r'^(?:t|f)$'
                ]
            },
            'id': {
                'name_patterns': [
                    r'\b(?:id|identifier|key|code|number|num|no|#)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:id|identifier|key|code|number|num|no))\b',  # customer_id, order_id, etc.
                    r'\b(?:primary|foreign|unique)[\s_-]*(?:key|id|identifier)\b'
                ],
                'value_patterns': [
                    r'^\d{4,}$',  # 4+ digit numbers
                    r'^[A-Za-z]{1,3}[-_]?\d{3,}$',  # AB123, A-123
                    r'^\d{3,}[-_]?[A-Za-z]{1,3}$',  # 123AB, 123-A
                    r'^[A-Fa-f0-9]{8}-?[A-Fa-f0-9]{4}-?[A-Fa-f0-9]{4}-?[A-Fa-f0-9]{4}-?[A-Fa-f0-9]{12}$'  # UUID
                ]
            },
            'name': {
                'name_patterns': [
                    r'\b(?:name|title|label|caption|heading)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:name|title|label))\b',  # product_name, category_name, etc.
                    r'\b(?:full|display|short|long)[\s_-]*(?:name|title|label)\b'
                ]
            },
            'description': {
                'name_patterns': [
                    r'\b(?:description|desc|details|info|information|overview|summary)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:description|desc|details))\b',  # product_desc, item_details, etc.
                    r'\b(?:short|long|brief|full|detailed)[\s_-]*(?:description|desc|details)\b'
                ]
            },
            'category': {
                'name_patterns': [
                    r'\b(?:category|cat|type|group|class|classification|segment|tier|level)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:category|type|group|class))\b',  # product_category, item_type, etc.
                    r'\b(?:main|primary|secondary|sub)[\s_-]*(?:category|cat|type|group|class)\b'
                ]
            },
            'status': {
                'name_patterns': [
                    r'\b(?:status|state|condition|stage|phase|position|progress)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:status|state))\b',  # order_status, account_state, etc.
                    r'\b(?:current|previous|next|final)[\s_-]*(?:status|state|stage|phase)\b'
                ],
                'value_patterns': [
                    r'^(?:active|inactive|pending|completed|canceled|on[\s_-]*hold|new|processing|confirmed|shipped)$',
                    r'^(?:A|I|P|C|X|H|N|S)$'
                ]
            },
            'url': {
                'name_patterns': [
                    r'\b(?:url|uri|link|hyperlink|web[\s_-]*address|site)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:url|uri|link))\b',  # product_url, image_link, etc.
                    r'\b(?:http|https|ftp|website|webpage)[\s_-]*(?:link|url|address)?\b'
                ],
                'value_patterns': [
                    r'^(?:https?|ftp)://[^\s/$.?#].[^\s]*$',
                    r'^www\.[^\s/$.?#].[^\s]*$'
                ]
            },
            'email': {
                'name_patterns': [
                    r'\b(?:email|e[\s_-]*mail|mail)\b',
                    r'\b(?:[a-z]+[\s_-]*(?:email|mail))\b',  # customer_email, contact_mail, etc.
                    r'\b(?:electronic|digital)[\s_-]*(?:mail|mailbox|address)\b'
                ],
                'value_patterns': [
                    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                ]
            },
            'phone': {
                'name_patterns': [
                    r'\b(?:phone|telephone|mobile|cell|contact)[\s_-]*(?:number|no|#)?\b',
                    r'\b(?:[a-z]+[\s_-]*(?:phone|telephone|mobile|cell))\b',  # customer_phone, contact_mobile, etc.
                    r'\b(?:work|home|business|fax)[\s_-]*(?:phone|telephone|number)\b'
                ],
                'value_patterns': [
                    r'^\+?[\d\s-\(\).]{7,}$'
                ]
            }
        }
    
    def _initialize_customer_patterns(self):
        """Initialize customer-related patterns"""
        return {
            'customer_id': {
                'name_patterns': [
                    r'\b(?:customer|client|buyer|account|contact)[\s_-]*(?:id|number|code|no|identifier|key)\b',
                    r'\b(?:cust|clnt|acct)[\s_-]*(?:id|code|num|no)\b',
                    r'\bcust[\s_-]*(?:#|no|num|code)\b'
                ],
                'value_patterns': [
                    r'^[A-Z]{1,3}-\d{4,10}$',
                    r'^CUST\d{4,}$',
                    r'^\d{5,10}$'  # Simple numeric customer IDs
                ]
            },
            'customer_name': {
                'name_patterns': [
                    r'\b(?:customer|client|buyer|account)[\s_-]*(?:name|title|fullname)\b',
                    r'\b(?:first|last|given|family|middle|full)[\s_-]*name\b',
                    r'\b(?:fname|lname|fullname)\b'
                ],
                'value_patterns': [
                    r'^[A-Z][a-z]+\s+[A-Z][a-z]+$',  # First Last format
                    r'^[A-Z][a-z]+,\s+[A-Z][a-z]+$'  # Last, First format
                ]
            },
            'customer_segment': {
                'name_patterns': [
                    r'\b(?:customer|client|buyer|account)[\s_-]*(?:segment|tier|group|level|type|category|cohort)\b',
                    r'\b(?:loyalty|vip|premium|gold|silver|bronze)[\s_-]*(?:tier|level|status|group)\b',
                    r'\b(?:segment|segmentation|cluster)[\s_-]*(?:id|code|name|group|value)?\b'
                ],
                'value_patterns': [
                    r'^(?:premium|standard|basic|gold|silver|bronze|platinum|vip|regular)$',
                    r'^tier[\s_-]?\d$',
                    r'^[A-D]$'  # Simple letter-based segments
                ]
            },
            'customer_acquisition': {
                'name_patterns': [
                    r'\b(?:customer|client|lead|prospect)[\s_-]*(?:acquisition|source|origin|channel)\b',
                    r'\b(?:acquisition|conversion|signup|registration|join|onboarding)[\s_-]*(?:source|channel|medium|campaign)\b',
                    r'\b(?:lead|referral|traffic)[\s_-]*source\b'
                ],
                'value_patterns': [
                    r'^(?:web|email|social|referral|organic|paid|direct|offline|store|call|event)$',
                    r'^(?:facebook|google|twitter|linkedin|instagram|tiktok|youtube)$'
                ]
            },
            'customer_ltv': {
                'name_patterns': [
                    r'\b(?:customer|client|buyer|account)[\s_-]*(?:ltv|lifetime[\s_-]*value|clv|value)\b',
                    r'\b(?:predicted|forecasted|estimated|projected)[\s_-]*(?:ltv|lifetime[\s_-]*value|clv)\b',
                    r'\b(?:ltv|clv|cltv)[\s_-]*(?:value|amount|score)?\b'
                ]
            },
            'customer_contact': {
                'name_patterns': [
                    r'\b(?:email|e-?mail|mail)[\s_-]*(?:address)?\b',
                    r'\b(?:phone|telephone|mobile|cell)[\s_-]*(?:number|no)?\b',
                    r'\bcontact[\s_-]*(?:details|info)\b'
                ],
                'value_patterns': {
                    'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                    'phone': r'^\+?[\d\s-\(\).]{7,}$'
                }
            },
            'customer_address': {
                'name_patterns': [
                    r'\b(?:billing|shipping|mailing|delivery|home|work|office)[\s_-]*address\b',
                    r'\b(?:street|avenue|road|boulevard|lane|plaza|building|apt|suite)[\s_-]*(?:address|number|no)?\b',
                    r'\b(?:address|addr)[\s_-]*(?:line[\s_-]*[1-3]|1|2|3)\b'
                ]
            },
            'customer_status': {
                'name_patterns': [
                    r'\b(?:customer|client|account)[\s_-]*(?:status|state|condition|standing)\b',
                    r'\b(?:active|inactive|suspended|pending|canceled|terminated|churned)[\s_-]*(?:status|flag|indicator)?\b',
                    r'\b(?:status|state)[\s_-]*(?:code|value|indicator)?\b'
                ],
                'value_patterns': [
                    r'^(?:active|inactive|pending|suspended|closed|terminated|on[\s_-]*hold|new|verified)$',
                    r'^(?:0|1|Y|N|A|I|P)$'  # Common status codes
                ]
            },
            'customer_age': {
                'name_patterns': [
                    r'\b(?:customer|client|user|member|account)[\s_-]*(?:age|years|yrs)\b',
                    r'\b(?:age|years|yrs)[\s_-]*(?:old)?\b',
                    r'\b(?:birth|dob|birth[\s_-]*date)[\s_-]*(?:date|day|year)?\b'
                ],
                'value_patterns': [
                    r'^(?:1[89]|[2-9][0-9])$',  # Ages 18-99
                    r'^\d{1,2}/\d{1,2}/\d{4}$',  # MM/DD/YYYY
                    r'^\d{4}-\d{2}-\d{2}$'  # YYYY-MM-DD
                ]
            },
            'customer_gender': {
                'name_patterns': [
                    r'\b(?:gender|sex|male|female)\b',
                    r'\b(?:customer|client|user|member|account)[\s_-]*(?:gender|sex)\b'
                ],
                'value_patterns': [
                    r'^(?:M|F|Male|Female|man|woman|non[\s_-]*binary|other)$',
                    r'^(?:m|f|male|female)$'
                ]
            },
            'customer_purchase_frequency': {
                'name_patterns': [
                    r'\b(?:customer|client|buyer|account)[\s_-]*(?:purchase|order|transaction)[\s_-]*(?:frequency|rate|count|number)\b',
                    r'\b(?:purchase|order|buy|transaction)[\s_-]*(?:frequency|rate|cadence|pattern|cycle)\b',
                    r'\b(?:frequency|times|count)[\s_-]*(?:of|per)[\s_-]*(?:purchase|order|transaction)\b'
                ]
            },
            'customer_churn_risk': {
                'name_patterns': [
                    r'\b(?:customer|client|account|churn|attrition)[\s_-]*(?:risk|score|probability|likelihood|potential)\b',
                    r'\b(?:churn|attrition|loss|cancellation|defection)[\s_-]*(?:risk|score|rate|prediction|probability)\b',
                    r'\b(?:risk|likelihood|probability)[\s_-]*(?:of|to)[\s_-]*(?:churn|cancel|leave|attrition)\b'
                ],
                'value_patterns': [
                    r'^(?:high|medium|low|H|M|L|critical|warning|safe)$',
                    r'^(?:0|[0-9]\.[0-9]+|[0-9]{1,2}|100)%?$'  # 0-100% or 0.0-1.0
                ]
            }
        }
    
    def _initialize_time_patterns(self):
        """Initialize time-related patterns"""
        return {
            'time_of_day': {
                'name_patterns': [
                    r'\b(?:time|hour|minute|second)[\s_-]*(?:of[\s_-]*day)?\b',
                    r'\b(?:timestamp|datetime|datetimeoffset)\b',
                    r'\b(?:clock|wall)[\s_-]*time\b'
                ],
                'value_patterns': [
                    r'^\d{1,2}:\d{2}(:\d{2})?([aApP][mM])?$',  # 3:30PM, 15:45, 9:20:15
                    r'^\d{1,2}[aApP][mM]$'  # 3PM, 11am
                ]
            },
            'duration': {
                'name_patterns': [
                    r'\b(?:duration|elapsed|length|span|interval|period)[\s_-]*(?:time|minutes|seconds|hours)?\b',
                    r'\b(?:time[\s_-]*spent|time[\s_-]*elapsed|time[\s_-]*taken)\b',
                    r'\b(?:session|call|visit|stay|usage)[\s_-]*(?:length|duration|time)\b'
                ],
                'value_patterns': [
                    r'^\d+:?\d*:?\d*$',  # 30, 1:30, 2:45:30
                    r'^\d+[\s_-]*(?:ms|sec|min|hr|day|wk|mo|yr)$'  # 30sec, 45min, 2hr
                ]
            },
            'delivery_time': {
                'name_patterns': [
                    r'\b(?:delivery|shipping|arrival|fulfillment|transit)[\s_-]*(?:time|date|timeframe|window|period|schedule)\b',
                    r'\b(?:estimated|actual|scheduled|promised|target)[\s_-]*(?:delivery|arrival|ship)[\s_-]*(?:date|time)\b',
                    r'\b(?:eta|etd|ata|atd)[\s_-]*(?:date|time)?\b'  # Estimated/Actual Time of Arrival/Departure
                ]
            },
            'lead_time': {
                'name_patterns': [
                    r'\b(?:lead|lag|processing|handling|turnaround|response)[\s_-]*time\b',
                    r'\b(?:time[\s_-]*to[\s_-]*(?:process|fulfill|complete|respond|ship|deliver))\b',
                    r'\b(?:sla|service[\s_-]*level[\s_-]*agreement)[\s_-]*(?:time|hours|days)?\b'
                ]
            },
            'time_zone': {
                'name_patterns': [
                    r'\b(?:time[\s_-]*zone|tz|timezone|time[\s_-]*offset|utc[\s_-]*offset)\b',
                    r'\b(?:local|server|system|user)[\s_-]*(?:time[\s_-]*zone|tz)\b'
                ],
                'value_patterns': [
                    r'^(?:UTC|GMT|EST|CST|MST|PST|EDT|CDT|MDT|PDT)[\+\-]?\d*$',
                    r'^[\+\-]\d{1,2}(?::\d{2})?$'  # +8, -5:30
                ]
            },
            'frequency': {
                'name_patterns': [
                    r'\b(?:frequency|interval|periodicity|recurrence|cycle|cadence)\b',
                    r'\b(?:daily|weekly|monthly|quarterly|yearly|annual)[\s_-]*(?:frequency|occurrence|schedule)?\b',
                    r'\b(?:times[\s_-]*per[\s_-]*(?:day|week|month|year|quarter))\b'
                ],
                'value_patterns': [
                    r'^(?:daily|weekly|biweekly|monthly|quarterly|annually|hourly|minutely)$',
                    r'^(?:every|once)[\s_-]+(?:day|week|month|year|hour|minute|second)$',
                    r'^\d+[\s_-]*(?:ms|sec|min|hr|day|wk|mo|yr)$'
                ]
            },
            'datetime_components': {
                'name_patterns': [
                    r'\b(?:year|yr|yyyy|fiscal[\s_-]*year|calendar[\s_-]*year)\b',
                    r'\b(?:month|mon|mm|quarter|qtr|season)\b',
                    r'\b(?:day|dy|dd|weekday|dow|dom)\b',
                    r'\b(?:hour|hr|minute|min|second|sec)\b'
                ],
                'value_patterns': {
                    'year': r'^(?:19|20)\d{2}$',  # 1900-2099
                    'month': r'^(?:0?[1-9]|1[0-2])$|^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$',
                    'day': r'^(?:0?[1-9]|[12][0-9]|3[01])$',
                    'hour': r'^(?:[01]?[0-9]|2[0-3])$',
                    'minute': r'^(?:[0-5]?[0-9])$',
                }
            },
            'week': {
                'name_patterns': [
                    r'\b(?:week|wk|week[\s_-]*number|week[\s_-]*no|weeknum)\b',
                    r'\b(?:iso|calendar|fiscal)[\s_-]*week\b',
                    r'\b(?:woy|wow|ww)[\s_-]*(?:number|no)?\b'  # Week of year, Week over week, Work week
                ],
                'value_patterns': [
                    r'^(?:[0-9]|[1-4][0-9]|5[0-3])$',  # 0-53
                    r'^W(?:[0-9]|[1-4][0-9]|5[0-3])$'  # W1-W53
                ]
            }
        }
    
    def _initialize_location_patterns(self):
        """Initialize geographic/location patterns"""
        return {
            'address': {
                'name_patterns': [
                    r'\b(?:address|addr|location|street|ave|blvd|road|rd)[\s_-]*(?:line)?[\s_-]*(?:1|2|3|one|two|three)?\b',
                    r'\b(?:billing|shipping|mailing|delivery|home|work|business)[\s_-]*address\b',
                    r'\b(?:street|building|location)[\s_-]*(?:address|number|name)\b'
                ]
            },
            'city': {
                'name_patterns': [
                    r'\b(?:city|town|municipality|borough|village|suburb|settlement|urban[\s_-]*area)\b',
                    r'\b(?:billing|shipping|mailing|delivery|home|work|business)[\s_-]*city\b',
                    r'\b(?:city|town)[\s_-]*(?:name|code)?\b'
                ]
            },
            'state_province': {
                'name_patterns': [
                    r'\b(?:state|province|county|region|district|territory|prefecture)\b',
                    r'\b(?:billing|shipping|mailing|delivery|home|work|business)[\s_-]*(?:state|province)\b',
                    r'\b(?:state|province)[\s_-]*(?:name|code|abbr|abbreviation)?\b'
                ],
                'value_patterns': [
                    r'^[A-Z]{2}$',  # US state codes: NY, CA, etc.
                    r'^(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)$'  # Specific US states
                ]
            },
            'postal_code': {
                'name_patterns': [
                    r'\b(?:zip|postal|post)[\s_-]*(?:code)?\b',
                    r'\b(?:billing|shipping|mailing|delivery|home|work|business)[\s_-]*(?:zip|postal|post)[\s_-]*(?:code)?\b'
                ],
                'value_patterns': [
                    r'^\d{5}(?:-\d{4})?$',  # US ZIP: 12345 or 12345-6789
                    r'^[A-Z]\d[A-Z][\s-]?\d[A-Z]\d$'  # Canadian postal code: A1A 1A1
                ]
            },
            'country': {
                'name_patterns': [
                    r'\b(?:country|nation|land|territory|commonwealth|republic|kingdom)\b',
                    r'\b(?:billing|shipping|mailing|delivery|home|work|business)[\s_-]*country\b',
                    r'\b(?:country|nation)[\s_-]*(?:name|code|iso)?\b'
                ],
                'value_patterns': [
                    r'^(?:US|USA|CA|CAN|UK|GB|AU|DE|FR|JP|CN|IN|BR)$',  # Common ISO codes
                    r'^[A-Z]{2}$',  # ISO 2-letter codes
                    r'^[A-Z]{3}$'   # ISO 3-letter codes
                ]
            },
            'region': {
                'name_patterns': [
                    r'\b(?:region|area|zone|district|sector|territory|jurisdiction)\b',
                    r'\b(?:sales|service|delivery|market)[\s_-]*(?:region|area|zone|territory)\b',
                    r'\b(?:region|zone)[\s_-]*(?:name|code|id)?\b'
                ]
            },
            'geo_coordinates': {
                'name_patterns': [
                    r'\b(?:latitude|lat|longitude|long|lon|lng|coords|coordinates|geo|gps|position)\b',
                    r'\b(?:lat|latitude)[\s_-]*(?:value|coordinate|position|degrees)?\b',
                    r'\b(?:lon|long|longitude)[\s_-]*(?:value|coordinate|position|degrees)?\b'
                ],
                'value_patterns': {
                    'latitude': r'^-?(?:90|[1-8]?[0-9](?:\.\d+)?)$',
                    'longitude': r'^-?(?:180|1[0-7][0-9]|[1-9]?[0-9](?:\.\d+)?)$'
                }
            },
            'store_location': {
                'name_patterns': [
                    r'\b(?:store|shop|outlet|branch|location|site|dealer|franchise)[\s_-]*(?:id|code|number|location|address)?\b',
                    r'\b(?:retail|warehouse|distribution|pickup|collection)[\s_-]*(?:location|site|center|facility)\b'
                ],
                'value_patterns': [
                    r'^[A-Z]{1,2}[-_]?\d{2,4}$',  # Store codes like NY-123, TX12
                    r'^ST\d{3,}$',  # ST123, ST4567
                    r'^\d{3,5}$'    # Simple numeric store IDs
                ]
            }
        }
    
    def _initialize_marketing_patterns(self):
        """Initialize marketing and campaign patterns"""
        return {
            'campaign': {
                'name_patterns': [
                    r'\b(?:campaign|promotion|promo|initiative|drive|marketing)[\s_-]*(?:id|code|name|number|no|identifier)?\b',
                    r'\b(?:advertisement|ad|advert)[\s_-]*(?:campaign|id|code|name|number)?\b',
                    r'\b(?:camp|cmpgn)[\s_-]*(?:id|code|num|no)?\b'
                ]
            },
            'channel': {
                'name_patterns': [
                    r'\b(?:channel|medium|platform|source|touchpoint|outlet)\b',
                    r'\b(?:marketing|sales|distribution|communication|acquisition)[\s_-]*(?:channel|source|medium)\b',
                    r'\b(?:utm|tracking)[\s_-]*(?:source|medium|channel)\b'
                ],
                'value_patterns': [
                    r'^(?:web|email|social|print|tv|radio|direct|mail|store|online|offline|mobile|app|referral)$',
                    r'^(?:facebook|twitter|instagram|linkedin|youtube|google|pinterest|tiktok)$'
                ]
            },
            'promotion': {
                'name_patterns': [
                    r'\b(?:promotion|promo|offer|deal|discount|special)[\s_-]*(?:id|code|name|type)?\b',
                    r'\b(?:coupon|voucher|rebate)[\s_-]*(?:code|id|number|amount|value|percent)?\b',
                    r'\b(?:promo|discount)[\s_-]*(?:rate|amount|value|percentage|level|tier)?\b'
                ]
            },
            'segment': {
                'name_patterns': [
                    r'\b(?:segment|segmentation|audience|cohort|group|cluster|category)\b',
                    r'\b(?:market|customer|user|buyer|client|visitor)[\s_-]*(?:segment|group|cluster|category|cohort)\b',
                    r'\b(?:demographic|psychographic|behavioral)[\s_-]*(?:segment|group|cluster|category)?\b'
                ]
            },
            'conversion': {
                'name_patterns': [
                    r'\b(?:conversion|click|action|engagement|interaction|event)[\s_-]*(?:rate|percentage|count|number|value)?\b',
                    r'\b(?:bounce|exit|abandonment)[\s_-]*rate\b',
                    r'\b(?:ctr|cvr|cpa|cpc|cpm)[\s_-]*(?:value|rate|amount)?\b'  # Click-through rate, Conversion rate, etc.
                ]
            },
            'attribution': {
                'name_patterns': [
                    r'\b(?:attribution|contribution|credit|source|origin|influence)\b',
                    r'\b(?:first|last|multi|linear|time|position|touch)[\s_-]*(?:touch|click|interaction|attribution|credit)\b',
                    r'\b(?:attribution|contribution)[\s_-]*(?:model|method|approach|algorithm|rule|logic)\b'
                ],
                'value_patterns': [
                    r'^(?:first|last|linear|time|position|multi|data)[\s_-]*(?:touch|click|interaction|attribution)$',
                    r'^(?:direct|organic|referral|social|email|paid|affiliate|partner)$'
                ]
            },
            'marketing_cost': {
                'name_patterns': [
                    r'\b(?:marketing|advertising|promotion|campaign)[\s_-]*(?:cost|expense|spend|budget|investment)\b',
                    r'\b(?:ad|campaign|promotion)[\s_-]*(?:spend|cost|expense|budget)\b',
                    r'\b(?:cac|cpa|cpc|cpm|cpp)[\s_-]*(?:cost|value|amount|rate)?\b'  # Cost per acquisition, Cost per click, etc.
                ]
            },
            'marketing_roi': {
                'name_patterns': [
                    r'\b(?:marketing|advertising|promotion|campaign)[\s_-]*(?:roi|return|roas|performance|efficiency)\b',
                    r'\b(?:roi|roas|romi|return)[\s_-]*(?:on|of)[\s_-]*(?:marketing|advertising|ad|campaign|promotion)[\s_-]*(?:spend|investment|expense)?\b',
                    r'\b(?:campaign|ad|promotion)[\s_-]*(?:roi|return|performance|results|success|effectiveness)\b'
                ],
                'value_patterns': [
                    r'^-?\d+(?:\.\d+)?%?$',  # Numeric ROI (possibly with percentage sign)
                    r'^-?\d+(?:\.\d+)?[xX]$'  # ROI in format like "3.5x"
                ]
            }
        }
    
    def _initialize_operational_patterns(self):
        """Initialize operational/fulfillment patterns"""
        return {
            'order_status': {
                'name_patterns': [
                    r'\b(?:order|shipment|delivery|fulfillment|transaction)[\s_-]*(?:status|state|condition|stage|phase)\b',
                    r'\b(?:current|latest|updated|tracking)[\s_-]*(?:status|state|condition)\b',
                    r'\b(?:status|state)[\s_-]*(?:code|value|indicator|flag)?\b'
                ],
                'value_patterns': [
                    r'^(?:new|pending|processing|shipped|delivered|completed|cancelled|returned|on[-\s_]*hold|back[-\s_]*ordered)$',
                    r'^(?:N|P|S|D|C|X|R|H|B)$'  # Status codes
                ]
            },
            'shipping_method': {
                'name_patterns': [
                    r'\b(?:shipping|delivery|transport|carrier|shipment|fulfillment)[\s_-]*(?:method|type|mode|service|option|provider|carrier|company)?\b',
                    r'\b(?:express|standard|overnight|priority|ground|air|freight)[\s_-]*(?:shipping|delivery|service)?\b',
                    r'\b(?:ship|delivery)[\s_-]*(?:via|by|through|method|type|mode)\b'
                ],
                'value_patterns': [
                    r'^(?:standard|express|overnight|priority|ground|air|2[-\s_]*day|next[-\s_]*day)$',
                    r'^(?:fedex|ups|usps|dhl|amazon|royal[-\s_]*mail|canada[-\s_]*post)$'
                ]
            },
            'tracking_id': {
                'name_patterns': [
                    r'\b(?:tracking|shipment|package|parcel|delivery)[\s_-]*(?:id|number|code|identifier|reference|no)\b',
                    r'\b(?:track|trace)[\s_-]*(?:id|no|number|code)\b',
                    r'\b(?:waybill|airway[-\s_]*bill|bill[-\s_]*of[-\s_]*lading)[\s_-]*(?:number|no|id|code)?\b'
                ],
                'value_patterns': [
                    r'^[0-9]{8,15}$',  # Basic numeric tracking
                    r'^[A-Z]{2}[0-9]{9}[A-Z]{2}$',  # USPS format
                    r'^1Z[A-Z0-9]{16}$'  # UPS format
                ]
            },
            'fulfillment_center': {
                'name_patterns': [
                    r'\b(?:fulfillment|distribution|warehouse|storage|inventory|logistics)[\s_-]*(?:center|facility|location|building|site|hub|depot)\b',
                    r'\b(?:fc|dc|wh)[\s_-]*(?:id|code|number|name|location)?\b',
                    r'\b(?:shipping|fulfillment|shipping)[\s_-]*(?:from|origin|source|location)\b'
                ]
            },
            'inventory_status': {
                'name_patterns': [
                    r'\b(?:inventory|stock|supply|availability|quantity)[\s_-]*(?:status|level|state|condition|position)\b',
                    r'\b(?:in[-\s_]*stock|out[-\s_]*of[-\s_]*stock|available|unavailable|backorder)[\s_-]*(?:status|flag|indicator)?\b',
                    r'\b(?:stock|inventory)[\s_-]*(?:on[-\s_]*hand|available|reserved|allocated|committed)\b'
                ],
                'value_patterns': [
                    r'^(?:in[-\s_]*stock|out[-\s_]*of[-\s_]*stock|low[-\s_]*stock|available|unavailable|backorder)$',
                    r'^(?:1|0|Y|N|A|U|L|B)$'  # Status codes (1/0, Yes/No, etc.)
                ]
            },
            'batch_number': {
                'name_patterns': [
                    r'\b(?:batch|lot|production|manufacturing)[\s_-]*(?:number|no|id|code|identifier)\b',
                    r'\b(?:batch|lot)[\s_-]*(?:qty|quantity|volume|amount|count)?\b',
                    r'\b(?:production|process|manufacturing)[\s_-]*(?:run|sequence|series)[\s_-]*(?:id|no|number)?\b'
                ],
                'value_patterns': [
                    r'^[A-Z]{1,3}[0-9]{4,8}$',  # BA12345
                    r'^[0-9]{4,8}[A-Z]{1,3}$',  # 12345BA
                    r'^(?:B|L|P)-[0-9]{4,8}$'  # B-12345
                ]
            },
            'order_id': {
                'name_patterns': [
                    r'\b(?:order|purchase|transaction|invoice|sales|checkout)[\s_-]*(?:id|number|no|code|identifier|reference)\b',
                    r'\b(?:ord|po|pur|inv)[\s_-]*(?:id|no|number|code|#)\b',
                    r'\b(?:confirmation|receipt)[\s_-]*(?:number|no|id|code)\b'
                ],
                'value_patterns': [
                    r'^ORD-?\d{5,10}$',  # ORD12345, ORD-12345
                    r'^PO-?\d{5,10}$',   # PO12345, PO-12345
                    r'^\d{5,10}$'        # Simple numeric order IDs
                ]
            },
            'order_line': {
                'name_patterns': [
                    r'\b(?:order|line|item)[\s_-]*(?:line|item|position|number|sequence|sequence[\s_-]*number)\b',
                    r'\b(?:line|item)[\s_-]*(?:in|on|of)[\s_-]*(?:order|invoice|receipt)\b',
                    r'\b(?:item|position|line)[\s_-]*(?:#|no|number)\b'
                ],
                'value_patterns': [
                    r'^\d{1,3}$',  # Simple line numbers (1, 2, 3, etc.)
                    r'^\d{1,3}\.\d{1,3}$'  # Hierarchical line numbers (1.1, 1.2, etc.)
                ]
            }
        }
    
    def _initialize_temporal_patterns(self):
        """Initialize temporal patterns (expanded date/time detection)"""
        return {
            'year': {
                'name_patterns': [
                    r'\b(?:year|yr|yyyy|fiscal[\s_-]*year|fy|annual|calendar[\s_-]*year)\b',
                    r'\b(?:year|yr)[\s_-]*(?:of|in)[\s_-]*(?:order|purchase|transaction|sale|report)\b'
                ],
                'value_patterns': [
                    r'^(?:19|20)\d{2}$',  # 1900-2099
                    r'^\d{2}$',  # 2-digit years (22, 23, etc.)
                    r'^FY\d{2}(?:\d{2})?$'  # Fiscal year formats (FY22, FY2022)
                ]
            },
            'month': {
                'name_patterns': [
                    r'\b(?:month|mon|mm|mo|mnth)\b',
                    r'\b(?:month|mon)[\s_-]*(?:of|in)[\s_-]*(?:order|purchase|transaction|sale|year|report)\b',
                    r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(?:uary|ruary|ch|il|ust|tember|ober|ember)?\b'
                ],
                'value_patterns': [
                    r'^(?:0?[1-9]|1[0-2])$',  # 1-12 (or 01-12)
                    r'^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$',  # Three-letter abbreviations
                    r'^(?:January|February|March|April|May|June|July|August|September|October|November|December)$'  # Full names
                ]
            },
            'quarter': {
                'name_patterns': [
                    r'\b(?:quarter|qtr|q[1-4]|fiscal[\s_-]*quarter)\b',
                    r'\b(?:quarter|qtr)[\s_-]*(?:[1-4]|one|two|three|four)\b',
                    r'\b(?:q|quarter)[\s_-]*(?:of|in)[\s_-]*(?:year|fiscal|fy)\b'
                ],
                'value_patterns': [
                    r'^[Qq][1-4]$',  # Q1, Q2, Q3, Q4 (or q1, q2, q3, q4)
                    r'^[1-4]$',  # Simple 1, 2, 3, 4
                    r'^(?:first|second|third|fourth)[\s_-]*(?:quarter|qtr)$'  # Text formats
                ]
            },
            'day_of_week': {
                'name_patterns': [
                    r'\b(?:day[\s_-]*of[\s_-]*week|weekday|dow|day[\s_-]*name)\b',
                    r'\b(?:mon|tue|wed|thu|fri|sat|sun)(?:day|\.)?[\s_-]*(?:indicator|flag)?\b'
                ],
                'value_patterns': [
                    r'^(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)$',
                    r'^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)$',
                    r'^[0-6]$'  # 0-6 representation (often 0=Sunday or 0=Monday)
                ]
            },
            'day_of_month': {
                'name_patterns': [
                    r'\b(?:day|dy|dd|dom|day[\s_-]*of[\s_-]*month)\b',
                    r'\b(?:day)[\s_-]*(?:in|of)[\s_-]*(?:month|order|transaction|report)\b'
                ],
                'value_patterns': [
                    r'^(?:[1-9]|[12][0-9]|3[01])$',  # 1-31
                    r'^(?:0[1-9]|[12][0-9]|3[01])$'  # 01-31
                ]
            },
            'day_of_year': {
                'name_patterns': [
                    r'\b(?:day[\s_-]*of[\s_-]*year|doy|julian[\s_-]*day|ordinal[\s_-]*day)\b',
                    r'\b(?:yearly|annual)[\s_-]*(?:day|date)[\s_-]*(?:number|no)?\b'
                ],
                'value_patterns': [
                    r'^(?:[1-9]|[1-9][0-9]|[12][0-9]{2}|3[0-5][0-9]|36[0-6])$',  # 1-366
                    r'^(?:00[1-9]|0[1-9][0-9]|[12][0-9]{2}|3[0-5][0-9]|36[0-6])$'  # 001-366
                ]
            },
            'hour': {
                'name_patterns': [
                    r'\b(?:hour|hr|hrs|hh)\b',
                    r'\b(?:hour)[\s_-]*(?:of|in)[\s_-]*(?:day|order|transaction|report)\b'
                ],
                'value_patterns': [
                    r'^(?:[0-9]|1[0-9]|2[0-3])$',  # 0-23
                    r'^(?:0[0-9]|1[0-9]|2[0-3])$',  # 00-23
                    r'^(?:[1-9]|1[0-2])[\s_-]*(?:am|pm|AM|PM)$'  # 12-hour format
                ]
            },
            'minute': {
                'name_patterns': [
                    r'\b(?:minute|min|mm)\b',
                    r'\b(?:minute)[\s_-]*(?:of|in)[\s_-]*(?:hour|time|timestamp)\b'
                ],
                'value_patterns': [
                    r'^(?:[0-9]|[1-5][0-9])$',  # 0-59
                    r'^(?:0[0-9]|[1-5][0-9])$'  # 00-59
                ]
            },
            'second': {
                'name_patterns': [
                    r'\b(?:second|sec|ss)\b',
                    r'\b(?:second)[\s_-]*(?:of|in)[\s_-]*(?:minute|time|timestamp)\b'
                ],
                'value_patterns': [
                    r'^(?:[0-9]|[1-5][0-9])$',  # 0-59
                    r'^(?:0[0-9]|[1-5][0-9])$'  # 00-59
                ]
            },
            'period': {
                'name_patterns': [
                    r'\b(?:period|time[\s_-]*period|date[\s_-]*range|time[\s_-]*range|time[\s_-]*frame|window)\b',
                    r'\b(?:reporting|analysis|accounting|sales|fiscal)[\s_-]*(?:period|timeframe|time[\s_-]*frame)\b',
                    r'\b(?:from|start|begin|to|end|through)[\s_-]*(?:date|time|period)\b'
                ]
            },
            'season': {
                'name_patterns': [
                    r'\b(?:season|seasonal|quarter|period)\b',
                    r'\b(?:spring|summer|fall|autumn|winter)[\s_-]*(?:season|period|quarter)?\b',
                    r'\b(?:holiday|shopping|peak|off[\s_-]*peak)[\s_-]*(?:season|period|time)\b'
                ],
                'value_patterns': [
                    r'^(?:spring|summer|fall|autumn|winter)$',
                    r'^(?:q[1-4]|quarter[\s_-]*[1-4])$',
                    r'^(?:holiday|peak|off[\s_-]*peak|high|low)[\s_-]*(?:season)?$'
                ]
            }
        }
    
    def _initialize_financial_patterns(self):
        """Initialize financial patterns (expanded sales/revenue detection)"""
        return {
            'revenue': {
                'name_patterns': [
                    r'\b(?:revenue|sales|income|proceeds|turnover|earnings)\b',
                    r'\b(?:gross|net|total|monthly|quarterly|annual|daily)[\s_-]*(?:revenue|sales|income|proceeds)\b',
                    r'\b(?:revenue|sales)[\s_-]*(?:amount|figure|number|total|value|volume)\b'
                ]
            },
            'price': {
                'name_patterns': [
                    r'\b(?:price|cost|rate|charge|fee)\b',
                    r'\b(?:unit|per[\s_-]*item|single|individual)[\s_-]*(?:price|cost|rate|charge|fee)\b',
                    r'\b(?:price|cost)[\s_-]*(?:per|each|unit|item)\b'
                ]
            },
            'discount': {
                'name_patterns': [
                    r'\b(?:discount|reduction|markdown|savings|sale|promotion)\b',
                    r'\b(?:price|cost)[\s_-]*(?:discount|reduction|markdown|adjustment|deduction)\b',
                    r'\b(?:discount|promo|coupon|voucher)[\s_-]*(?:amount|value|percentage|rate|code)\b'
                ],
                'value_patterns': [
                    r'^-?\d+(?:\.\d+)?%?$',  # Numeric discount (may include % sign)
                    r'^-?\d+(?:\.\d+)?[\s_-]*(?:off|discount)$'  # Format like "25 off"
                ]
            },
            'tax': {
                'name_patterns': [
                    r'\b(?:tax|vat|gst|sales[\s_-]*tax|use[\s_-]*tax)\b',
                    r'\b(?:tax|vat|gst)[\s_-]*(?:amount|value|rate|percentage|total)\b',
                    r'\b(?:federal|state|local|city|county|provincial)[\s_-]*tax\b'
                ],
                'value_patterns': [
                    r'^\d+(?:\.\d+)?%?$'  # Tax rate (may include % sign)
                ]
            },
            'shipping': {
                'name_patterns': [
                    r'\b(?:shipping|freight|delivery|transport|postage|handling)\b',
                    r'\b(?:shipping|freight|delivery|transport)[\s_-]*(?:cost|charge|fee|expense|amount)\b',
                    r'\b(?:shipping|delivery)[\s_-]*(?:rate|price|cost|charge)\b'
                ]
            },
            'total': {
                'name_patterns': [
                    r'\b(?:total|sum|grand[\s_-]*total|final|overall)\b',
                    r'\b(?:total|final|overall)[\s_-]*(?:amount|cost|price|value|charge|fee)\b',
                    r'\b(?:order|invoice|purchase|transaction)[\s_-]*(?:total|amount|value|sum)\b'
                ]
            },
            'refund': {
                'name_patterns': [
                    r'\b(?:refund|return|reimbursement|chargeback|credit|money[\s_-]*back)\b',
                    r'\b(?:refund|return|reimbursement)[\s_-]*(?:amount|value|total|sum)\b',
                    r'\b(?:refunded|returned|credited|reimbursed)[\s_-]*(?:amount|value|total|sum)\b'
                ]
            },
            'cost': {
                'name_patterns': [
                    r'\b(?:cost|expense|expenditure|outlay|spend|spending)\b',
                    r'\b(?:cost|purchase|wholesale|acquisition)[\s_-]*(?:price|amount|value)\b',
                    r'\b(?:unit|per[\s_-]*item|manufacturing|production)[\s_-]*cost\b'
                ]
            },
            'margin': {
                'name_patterns': [
                    r'\b(?:margin|markup|profit[\s_-]*margin|contribution[\s_-]*margin)\b',
                    r'\b(?:gross|net|operating|profit)[\s_-]*margin\b',
                    r'\b(?:margin|markup)[\s_-]*(?:rate|percentage|amount|value)\b'
                ],
                'value_patterns': [
                    r'^\d+(?:\.\d+)?%?$'  # Margin rate (may include % sign)
                ]
            },
            'profit': {
                'name_patterns': [
                    r'\b(?:profit|earning|gain|surplus|advantage|yield|return)\b',
                    r'\b(?:gross|net|operating|pre[\s_-]*tax|after[\s_-]*tax)[\s_-]*(?:profit|earnings|income)\b',
                    r'\b(?:profit|earning)[\s_-]*(?:amount|value|total|sum)\b'
                ]
            },
            'currency': {
                'name_patterns': [
                    r'\b(?:currency|money|denomination|tender|legal[\s_-]*tender)\b',
                    r'\b(?:currency|monetary)[\s_-]*(?:code|unit|type|sign|symbol)\b',
                    r'\b(?:payment|transaction)[\s_-]*currency\b'
                ],
                'value_patterns': [
                    r'^(?:USD|EUR|GBP|JPY|CNY|AUD|CAD|CHF|INR|BRL)$',  # Common ISO codes
                    r'^[A-Z]{3}$',  # Any 3-letter code
                    r'^(?:\$|€|£|¥|₹|₽|₩|₿)$'  # Currency symbols
                ]
            }
        }
        
    def _initialize_product_patterns(self):
        """Initialize product patterns (expanded product detection)"""
        return {
            'product_id': {
                'name_patterns': [
                    r'\b(?:product|item|article|good|sku|upc|ean|isbn)[\s_-]*(?:id|code|number|no|identifier|key)\b',
                    r'\b(?:prod|prd|itm|art)[\s_-]*(?:id|code|num|no)\b',
                    r'\b(?:bar|qr)[\s_-]*code\b'
                ],
                'value_patterns': [
                    r'^[A-Z]{1,3}[-_]?\d{3,}$',  # ABC123, A-123, etc.
                    r'^\d{3,}[-_]?[A-Z]{1,3}$',  # 123ABC, 123-A, etc.
                    r'^(?:\d{8}|\d{12}|\d{13}|\d{14})$'  # UPC/EAN/ISBN formats
                ]
            },
            'product_name': {
                'name_patterns': [
                    r'\b(?:product|item|article|good|merchandise)[\s_-]*(?:name|title|label|description|desc)\b',
                    r'\b(?:prod|prd|itm|art)[\s_-]*(?:name|desc)\b',
                    r'\bdescription\b'
                ]
            },
            'product_category': {
                'name_patterns': [
                    r'\b(?:product|item)[\s_-]*(?:category|cat|type|group|class|classification|family)\b',
                    r'\b(?:category|cat|type|group|class|classification|family)[\s_-]*(?:of[\s_-]*(?:product|item))?\b',
                    r'\b(?:main|primary|secondary|sub)[\s_-]*(?:category|cat)\b'
                ],
                'value_patterns': [
                    r'^(?:electronics|clothing|food|beverage|furniture|home|garden|tools|automotive|beauty|health|toys|sports|books|music)$'
                ]
            },
            'product_brand': {
                'name_patterns': [
                    r'\b(?:brand|make|manufacturer|vendor|supplier|producer|creator)\b',
                    r'\b(?:product|item)[\s_-]*(?:brand|make|manufacturer|vendor)\b',
                    r'\b(?:brand|make|manufacturer)[\s_-]*(?:name|id|code)\b'
                ]
            },
            'product_color': {
                'name_patterns': [
                    r'\b(?:color|colour|hue|shade|tint)\b',
                    r'\b(?:product|item)[\s_-]*(?:color|colour)\b',
                    r'\b(?:color|colour)[\s_-]*(?:name|code|value|option)\b'
                ],
                'value_patterns': [
                    r'^(?:red|blue|green|yellow|black|white|orange|purple|pink|brown|gray|grey|silver|gold)$',
                    r'^#[0-9A-Fa-f]{6}$',  # Hex color codes
                    r'^RGB\(\d{1,3},\s*\d{1,3},\s*\d{1,3}\)$'  # RGB format
                ]
            },
            'product_size': {
                'name_patterns': [
                    r'\b(?:size|dimension|measurement|width|height|length|depth|diameter)\b',
                    r'\b(?:product|item)[\s_-]*(?:size|dimension)\b',
                    r'\b(?:size|dimension)[\s_-]*(?:value|option|chart)\b'
                ],
                'value_patterns': [
                    r'^(?:xs|s|m|l|xl|xxl|xxxl)$',  # Clothing sizes
                    r'^(?:small|medium|large|x-large|extra[\s_-]*large)$',
                    r'^\d+(?:\.\d+)?[\s_-]*(?:mm|cm|m|in|ft|oz|lb|kg|g)$'  # Measurements with units
                ]
            },
            'product_material': {
                'name_patterns': [
                    r'\b(?:material|fabric|substance|composition|made[\s_-]*of|construction)\b',
                    r'\b(?:product|item)[\s_-]*(?:material|fabric|composition)\b',
                    r'\b(?:material|fabric)[\s_-]*(?:type|composition|blend)\b'
                ],
                'value_patterns': [
                    r'^(?:cotton|polyester|wool|silk|leather|metal|plastic|wood|glass|ceramic|paper|rubber)$',
                    r'^(?:\d+%\s*)?[a-zA-Z]+(?:\s*\d+%)?$'  # Material blends like "60% cotton 40% polyester"
                ]
            },
            'product_weight': {
                'name_patterns': [
                    r'\b(?:weight|mass|heaviness|lightness)\b',
                    r'\b(?:product|item|package|shipping)[\s_-]*weight\b',
                    r'\b(?:weight)[\s_-]*(?:value|grams|kilograms|pounds|ounces)\b'
                ],
                'value_patterns': [
                    r'^\d+(?:\.\d+)?[\s_-]*(?:g|kg|oz|lb|lbs)$'  # Weight with units
                ]
            },
            'product_availability': {
                'name_patterns': [
                    r'\b(?:availability|in[\s_-]*stock|stock[\s_-]*level|inventory[\s_-]*status)\b',
                    r'\b(?:product|item)[\s_-]*(?:availability|in[\s_-]*stock)\b',
                    r'\b(?:available|unavailable|in[\s_-]*stock|out[\s_-]*of[\s_-]*stock)[\s_-]*(?:status|flag|indicator)?\b'
                ],
                'value_patterns': [
                    r'^(?:in[\s_-]*stock|out[\s_-]*of[\s_-]*stock|available|unavailable|backordered|discontinued)$',
                    r'^(?:yes|no|y|n|true|false|1|0)$',
                    r'^\d+$'  # Quantity available
                ]
            },
            'product_sku': {
                'name_patterns': [
                    r'\b(?:sku|stock[\s_-]*keeping[\s_-]*unit|item[\s_-]*number|part[\s_-]*number)\b',
                    r'\b(?:product|item)[\s_-]*sku\b',
                    r'\b(?:sku)[\s_-]*(?:code|value|number|id)\b'
                ],
                'value_patterns': [
                    r'^[A-Z0-9]{3,}(?:[-_][A-Z0-9]+)*$',  # Common SKU formats
                    r'^[A-Z]{2,3}-\d{3,6}$'  # Format like AB-12345
                ]
            },
            'product_rating': {
                'name_patterns': [
                    r'\b(?:rating|review[\s_-]*score|star[\s_-]*rating|customer[\s_-]*rating|quality[\s_-]*score)\b',
                    r'\b(?:product|item)[\s_-]*(?:rating|score|stars)\b',
                    r'\b(?:average|overall)[\s_-]*(?:rating|score|stars)\b'
                ],
                'value_patterns': [
                    r'^[0-5](?:\.\d+)?$',  # 0-5 scale with decimals
                    r'^[0-9](?:\.\d+)?\/10$',  # x/10 format
                    r'^[1-5][\s_-]*stars?$'  # "4 stars" format
                ]
            },
            'product_price': {
                'name_patterns': [
                    r'\b(?:price|cost|retail[\s_-]*price|list[\s_-]*price|sale[\s_-]*price)\b',
                    r'\b(?:product|item)[\s_-]*(?:price|cost)\b',
                    r'\b(?:regular|special|sale|discount|promotional)[\s_-]*(?:price|cost)\b'
                ],
                'value_patterns': [
                    r'^\$\d+(?:\.\d{2})?$',  # $xx.xx format
                    r'^\d+(?:\.\d{2})?\s*(?:USD|EUR|GBP|JPY|CAD|AUD)$',  # xx.xx USD format
                    r'^(?:€|£|¥|₹|₽|₩)\d+(?:\.\d{2})?$'  # Other currency symbols
                ]
            },
            'product_url': {
                'name_patterns': [
                    r'\b(?:product|item)[\s_-]*(?:url|link|page|webpage)\b',
                    r'\b(?:url|link|web[\s_-]*address)[\s_-]*(?:to|for)[\s_-]*(?:product|item)\b',
                    r'\b(?:product|item)[\s_-]*(?:detail|landing)[\s_-]*(?:page|url|link)\b'
                ],
                'value_patterns': [
                    r'^https?://[^\s/$.?#].[^\s]*$',  # Standard URL format
                    r'^/[^\s]*$'  # Relative URL path
                ]
            },
            'product_image': {
                'name_patterns': [
                    r'\b(?:product|item)[\s_-]*(?:image|photo|picture|thumbnail|img)\b',
                    r'\b(?:image|photo|picture|thumbnail|img)[\s_-]*(?:of|for)[\s_-]*(?:product|item)\b',
                    r'\b(?:main|primary|default|featured)[\s_-]*(?:image|photo|picture)\b'
                ],
                'value_patterns': [
                    r'^https?://[^\s/$.?#].[^\s]*\.(?:jpg|jpeg|png|gif|webp|svg)$',  # Image URL
                    r'^/[^\s]*\.(?:jpg|jpeg|png|gif|webp|svg)$'  # Relative image path
                ]
            }
        }
    
    
    
        
    def _initialize_personnel_patterns(self):
        """Initialize personnel/employee patterns"""
        return {
            'employee_id': {
                'name_patterns': [
                    r'\b(?:employee|staff|personnel|worker|associate|team[\s_-]*member)[\s_-]*(?:id|number|code|no|identifier|key)\b',
                    r'\b(?:emp|ee)[\s_-]*(?:id|code|num|no)\b',
                    r'\b(?:employee|staff|personnel)[\s_-]*(?:#|no|num|code)\b'
                ],
                'value_patterns': [
                    r'^[A-Z]{1,2}-\d{4,6}$',  # E-12345, HR-123456
                    r'^EMP\d{4,}$',  # EMP1234
                    r'^\d{5,8}$'  # Simple numeric employee IDs
                ]
            },
            'employee_name': {
                'name_patterns': [
                    r'\b(?:employee|staff|personnel|worker|associate)[\s_-]*(?:name|fullname)\b',
                    r'\b(?:emp|ee)[\s_-]*(?:name|fullname)\b',
                    r'\b(?:worker|staff|personnel)[\s_-]*(?:name|fullname)\b'
                ],
                'value_patterns': [
                    r'^[A-Z][a-z]+\s+[A-Z][a-z]+$',  # First Last format
                    r'^[A-Z][a-z]+,\s+[A-Z][a-z]+$'  # Last, First format
                ]
            },
            'position': {
                'name_patterns': [
                    r'\b(?:position|title|job[\s_-]*title|role|job[\s_-]*role|designation)\b',
                    r'\b(?:employee|staff|personnel)[\s_-]*(?:position|title|role|designation)\b',
                    r'\b(?:job|work)[\s_-]*(?:title|position|function|role)\b'
                ]
            },
            'department': {
                'name_patterns': [
                    r'\b(?:department|dept|division|unit|group|team|function|business[\s_-]*unit)\b',
                    r'\b(?:employee|staff|personnel)[\s_-]*(?:department|dept|division|unit|group)\b',
                    r'\b(?:dept|division|unit)[\s_-]*(?:name|code|id)\b'
                ]
            },
            'manager': {
                'name_patterns': [
                    r'\b(?:manager|supervisor|leader|boss|head|chief|director|lead)\b',
                    r'\b(?:reports[\s_-]*to|managed[\s_-]*by|supervising|reporting[\s_-]*manager)\b',
                    r'\b(?:manager|supervisor|leader)[\s_-]*(?:id|name|code)\b'
                ]
            },
            'hire_date': {
                'name_patterns': [
                    r'\b(?:hire|hired|start|joining|employment|onboarding)[\s_-]*(?:date|day|on)\b',
                    r'\b(?:date)[\s_-]*(?:hired|joined|started|employed|of[\s_-]*hire)\b',
                    r'\b(?:employment|service)[\s_-]*(?:start|beginning|commencement)\b'
                ],
                'value_patterns': [
                    r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
                    r'^\d{1,2}/\d{1,2}/\d{2,4}$'  # M/D/YY or M/D/YYYY
                ]
            },
            'salary': {
                'name_patterns': [
                    r'\b(?:salary|wage|pay|compensation|earnings|income|remuneration)\b',
                    r'\b(?:annual|monthly|weekly|hourly|base)[\s_-]*(?:salary|wage|pay|compensation|earnings)\b',
                    r'\b(?:employee|staff|personnel)[\s_-]*(?:salary|wage|pay|compensation)\b'
                ],
                'value_patterns': [
                    r'^\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?$',  # $xx,xxx.xx format
                    r'^\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP)$',  # xx,xxx.xx USD format
                    r'^\d+(?:\.\d{2})?$'  # Simple numeric value
                ]
            },
            'employment_status': {
                'name_patterns': [
                    r'\b(?:employment|employee|staff|work|job)[\s_-]*(?:status|state|type|category)\b',
                    r'\b(?:status|state|standing)[\s_-]*(?:of|as)[\s_-]*(?:employee|employment|staff)\b',
                    r'\b(?:full[\s_-]*time|part[\s_-]*time|contract|temporary|permanent)[\s_-]*(?:status|flag|indicator)\b'
                ],
                'value_patterns': [
                    r'^(?:full[\s_-]*time|part[\s_-]*time|contract|temp|permanent|probation|intern|seasonal)$',
                    r'^(?:active|inactive|terminated|on[\s_-]*leave|suspended)$',
                    r'^(?:FT|PT|C|T|P|I|S)$'  # Abbreviated status codes
                ]
            },
            'performance_rating': {
                'name_patterns': [
                    r'\b(?:performance|evaluation|assessment|appraisal|review)[\s_-]*(?:rating|score|grade|result|review)\b',
                    r'\b(?:employee|staff|personnel)[\s_-]*(?:performance|evaluation|rating|score)\b',
                    r'\b(?:annual|quarterly|monthly)[\s_-]*(?:performance|evaluation|assessment|review)\b'
                ],
                'value_patterns': [
                    r'^[1-5]$',  # 1-5 scale
                    r'^[A-F]$',  # Letter grades
                    r'^(?:excellent|good|satisfactory|needs[\s_-]*improvement|unsatisfactory)$'
                ]
            },
            'tenure': {
                'name_patterns': [
                    r'\b(?:tenure|seniority|length[\s_-]*of[\s_-]*service|years[\s_-]*of[\s_-]*service|time[\s_-]*with[\s_-]*company)\b',
                    r'\b(?:employment|service)[\s_-]*(?:duration|length|period|tenure)\b',
                    r'\b(?:years|months)[\s_-]*(?:employed|with[\s_-]*company|of[\s_-]*employment)\b'
                ],
                'value_patterns': [
                    r'^\d+$',  # Simple numeric value (e.g., years)
                    r'^\d+\s*(?:years?|months?|days?)$',  # With time units
                    r'^\d+\s*years?,?\s*\d+\s*months?$'  # Combined format
                ]
            }
        }
    
    def _initialize_digital_patterns(self):
        """Initialize digital/online patterns"""
        return {
            'ip_address': {
                'name_patterns': [
                    r'\b(?:ip|internet[\s_-]*protocol)[\s_-]*(?:address|addr)\b',
                    r'\b(?:ipv4|ipv6)[\s_-]*(?:address|addr)?\b',
                    r'\b(?:client|server|user|visitor|customer)[\s_-]*(?:ip|internet[\s_-]*protocol)\b'
                ],
                'value_patterns': [
                    r'^(?:\d{1,3}\.){3}\d{1,3}$',  # IPv4 format
                    r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'  # Simple IPv6 format
                ]
            },
            'mac_address': {
                'name_patterns': [
                    r'\b(?:mac|media[\s_-]*access[\s_-]*control)[\s_-]*(?:address|addr)\b',
                    r'\b(?:physical|hardware|ethernet)[\s_-]*(?:address|addr)\b',
                    r'\b(?:device|network|nic)[\s_-]*(?:mac|address|identifier)\b'
                ],
                'value_patterns': [
                    r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$',  # Standard MAC format
                    r'^([0-9A-Fa-f]{2}){6}$'  # No separators
                ]
            },
            'user_agent': {
                'name_patterns': [
                    r'\b(?:user[\s_-]*agent|browser[\s_-]*signature|client[\s_-]*signature|browser[\s_-]*string)\b',
                    r'\b(?:http[\s_-]*user[\s_-]*agent|ua[\s_-]*string)\b',
                    r'\b(?:client|browser|visitor)[\s_-]*(?:identifier|id|signature)\b'
                ]
            },
            'device_type': {
                'name_patterns': [
                    r'\b(?:device|hardware|platform|system)[\s_-]*(?:type|category|class|kind)\b',
                    r'\b(?:client|user|visitor|customer)[\s_-]*(?:device|hardware|platform)\b',
                    r'\b(?:mobile|desktop|tablet|device)[\s_-]*(?:indicator|flag|type)\b'
                ],
                'value_patterns': [
                    r'^(?:desktop|mobile|tablet|smart[\s_-]*tv|console|wearable|iot)$',
                    r'^(?:android|ios|windows|mac|linux)$',
                    r'^(?:phone|computer|laptop|tv|watch)$'
                ]
            },
            'browser': {
                'name_patterns': [
                    r'\b(?:browser|web[\s_-]*browser|internet[\s_-]*browser)\b',
                    r'\b(?:browser|web[\s_-]*browser)[\s_-]*(?:type|name|version|info)\b',
                    r'\b(?:client|user|visitor)[\s_-]*(?:browser|web[\s_-]*client)\b'
                ],
                'value_patterns': [
                    r'^(?:chrome|firefox|safari|edge|opera|ie|internet[\s_-]*explorer)$',
                    r'^(?:chrome|firefox|safari|edge|opera|ie)\s*\d+(?:\.\d+){0,2}$'  # With version
                ]
            },
            'operating_system': {
                'name_patterns': [
                    r'\b(?:os|operating[\s_-]*system|platform|system[\s_-]*software)\b',
                    r'\b(?:client|user|visitor|device)[\s_-]*(?:os|operating[\s_-]*system|platform)\b',
                    r'\b(?:os|system|platform)[\s_-]*(?:version|name|type|info)\b'
                ],
                'value_patterns': [
                    r'^(?:windows|mac[\s_-]*os|ios|android|linux|ubuntu|chrome[\s_-]*os)$',
                    r'^(?:windows|mac[\s_-]*os|ios|android|linux)\s*\d+(?:\.\d+){0,2}$'  # With version
                ]
            },
            'screen_resolution': {
                'name_patterns': [
                    r'\b(?:screen|display|monitor|viewport|window)[\s_-]*(?:resolution|size|dimensions)\b',
                    r'\b(?:screen|display|monitor)[\s_-]*(?:width|height|pixels)\b',
                    r'\b(?:resolution|size)[\s_-]*(?:in[\s_-]*pixels|px)\b'
                ],
                'value_patterns': [
                    r'^\d{3,4}x\d{3,4}$',  # 1024x768 format
                    r'^\d{3,4}\s*[xX]\s*\d{3,4}$',  # 1024 x 768 format
                    r'^\d{3,4}\s*(?:px|pixels)?\s*[xX]\s*\d{3,4}\s*(?:px|pixels)?$'  # With "px"
                ]
            },
            'http_status': {
                'name_patterns': [
                    r'\b(?:http|status|response|request)[\s_-]*(?:status|code|result)\b',
                    r'\b(?:http|https)[\s_-]*(?:status|result|return)[\s_-]*(?:code|value)\b',
                    r'\b(?:server|api|web)[\s_-]*(?:response|status|result)[\s_-]*(?:code|value)\b'
                ],
                'value_patterns': [
                    r'^[1-5][0-9]{2}$',  # 3-digit codes (100-599)
                    r'^(?:200|404|500|403|401|301|302|400|504)$'  # Common status codes
                ]
            },
            'url_path': {
                'name_patterns': [
                    r'\b(?:url|uri|link|web)[\s_-]*(?:path|route|endpoint)\b',
                    r'\b(?:request|page|site|website|resource)[\s_-]*(?:path|route|url)\b',
                    r'\b(?:path|route|endpoint)[\s_-]*(?:url|uri|link|address)\b'
                ],
                'value_patterns': [
                    r'^/[^\s]*$',  # Starts with / and no spaces
                    r'^(?:/[a-zA-Z0-9_-]+)+/?$'  # Structured path
                ]
            }
        }
    
    def _initialize_statistical_patterns(self):
        """Initialize statistical/analytical patterns"""
        return {
            'mean': {
                'name_patterns': [
                    r'\b(?:mean|average|avg|arithmetic[\s_-]*mean)\b',
                    r'\b(?:mean|average|avg)[\s_-]*(?:value|score|rating|result)\b',
                    r'\b(?:mean|average|avg)[\s_-]*(?:of|for|across|over)\b'
                ]
            },
            'median': {
                'name_patterns': [
                    r'\b(?:median|middle[\s_-]*value|50th[\s_-]*percentile|p50)\b',
                    r'\b(?:median)[\s_-]*(?:value|score|rating|result)\b',
                    r'\b(?:median)[\s_-]*(?:of|for|across|over)\b'
                ]
            },
            'mode': {
                'name_patterns': [
                    r'\b(?:mode|most[\s_-]*frequent|most[\s_-]*common|modal[\s_-]*value)\b',
                    r'\b(?:mode)[\s_-]*(?:value|score|rating|result)\b',
                    r'\b(?:mode)[\s_-]*(?:of|for|across|over)\b'
                ]
            },
            'standard_deviation': {
                'name_patterns': [
                    r'\b(?:std|stdev|standard[\s_-]*deviation|sd|sigma)\b',
                    r'\b(?:std|stdev|standard[\s_-]*deviation)[\s_-]*(?:value|score|rating|result)\b',
                    r'\b(?:std|stdev|standard[\s_-]*deviation)[\s_-]*(?:of|for|across|over)\b'
                ]
            },
            'variance': {
                'name_patterns': [
                    r'\b(?:variance|var|squared[\s_-]*deviation)\b',
                    r'\b(?:variance|var)[\s_-]*(?:value|score|rating|result)\b',
                    r'\b(?:variance|var)[\s_-]*(?:of|for|across|over)\b'
                ]
            },
            'minimum': {
                'name_patterns': [
                    r'\b(?:minimum|min|lowest|smallest|least)\b',
                    r'\b(?:minimum|min|lowest)[\s_-]*(?:value|score|rating|result|observed)\b',
                    r'\b(?:minimum|min|lowest)[\s_-]*(?:of|for|across|over)\b'
                ]
            },
            'maximum': {
                'name_patterns': [
                    r'\b(?:maximum|max|highest|largest|greatest)\b',
                    r'\b(?:maximum|max|highest)[\s_-]*(?:value|score|rating|result|observed)\b',
                    r'\b(?:maximum|max|highest)[\s_-]*(?:of|for|across|over)\b'
                ]
            },
            'count': {
                'name_patterns': [
                    r'\b(?:count|n|num|number|frequency|occurrences|observations)\b',
                    r'\b(?:count|number|frequency)[\s_-]*(?:of|for|across|over)\b',
                    r'\b(?:total|overall)[\s_-]*(?:count|number|frequency)\b'
                ]
            },
            'sum': {
                'name_patterns': [
                    r'\b(?:sum|total|aggregate)\b',
                    r'\b(?:sum|total)[\s_-]*(?:value|score|amount|volume)\b',
                    r'\b(?:sum|total|aggregate)[\s_-]*(?:of|for|across|over)\b'
                ]
            },
            'range': {
                'name_patterns': [
                    r'\b(?:range|span|extent|spread|interval)\b',
                    r'\b(?:range|span|spread)[\s_-]*(?:value|score|amount|volume)\b',
                    r'\b(?:range|span|interval)[\s_-]*(?:of|for|across|over)\b'
                ]
            },
            'percentile': {
                'name_patterns': [
                    r'\b(?:percentile|quantile|p\d+|q\d+)\b',
                    r'\b(?:\d+(?:st|nd|rd|th))[\s_-]*(?:percentile|quantile)\b',
                    r'\b(?:percentile|quantile)[\s_-]*(?:value|score|rating|result)\b'
                ]
            },
            'correlation': {
                'name_patterns': [
                    r'\b(?:correlation|corr|relationship|association|covariance)\b',
                    r'\b(?:pearson|spearman|kendall)[\s_-]*(?:correlation|coefficient)\b',
                    r'\b(?:correlation|relationship)[\s_-]*(?:between|with|among)\b'
                ]
            },
            'significance': {
                'name_patterns': [
                    r'\b(?:significance|sig|p[\s_-]*value|alpha|confidence)\b',
                    r'\b(?:statistical|stat)[\s_-]*(?:significance|sig)\b',
                    r'\b(?:significance|sig|p[\s_-]*value)[\s_-]*(?:level|threshold|value)\b'
                ],
                'value_patterns': [
                    r'^0?\.\d+$',  # Decimal format (0.05)
                    r'^[1-9]\d*%$'  # Percentage format
                ]
            }
        }
    
    def _initialize_technical_patterns(self):
        """Initialize technical/system patterns"""
        return {
            'file_path': {
                'name_patterns': [
                    r'\b(?:file|path|directory|folder|location)[\s_-]*(?:path|location|name|address)\b',
                    r'\b(?:file|document|image|source|config)[\s_-]*(?:path|location|address)\b',
                    r'\b(?:path|location)[\s_-]*(?:to|of)[\s_-]*(?:file|document|image|source|config)\b'
                ],
                'value_patterns': [
                    r'^(?:/[^/]+)+/?$',  # Unix path
                    r'^(?:C:|D:|E:)(?:\\[^\\]+)+\\?$',  # Windows path
                    r'^(?:[A-Za-z]:)?(?:\\\\?|/)(?:[^\\/:*?"<>|\r\n]+\\\\?|/)*$'  # Generic path
                ]
            },
            'file_name': {
                'name_patterns': [
                    r'\b(?:file|document|image|source|config)[\s_-]*(?:name|title|label)\b',
                    r'\b(?:name|title|label)[\s_-]*(?:of)[\s_-]*(?:file|document|image|source|config)\b',
                    r'\b(?:file|document|image|filename)\b'
                ],
                'value_patterns': [
                    r'^[^\\/:*?"<>|\r\n]+\.[a-zA-Z0-9]{2,4}$'  # Name with extension
                ]
            },
            'file_extension': {
                'name_patterns': [
                    r'\b(?:file|document|image)[\s_-]*(?:extension|ext|type|format)\b',
                    r'\b(?:extension|ext|format)[\s_-]*(?:of)[\s_-]*(?:file|document|image)\b',
                    r'\b(?:file|document|image)[\s_-]*(?:suffix|extension)\b'
                ],
                'value_patterns': [
                    r'^\.?[a-zA-Z0-9]{2,4}$',  # .ext or ext format
                    r'^(?:txt|pdf|doc|docx|xls|xlsx|csv|jpg|jpeg|png|gif|html|xml|json)$'  # Common extensions without dot
                ]
            },
            'file_size': {
                'name_patterns': [
                    r'\b(?:file|document|image|attachment|upload)[\s_-]*(?:size|volume|bytes)\b',
                    r'\b(?:size|volume|bytes)[\s_-]*(?:of)[\s_-]*(?:file|document|image|attachment|upload)\b',
                    r'\b(?:file|document|image)[\s_-]*(?:size)[\s_-]*(?:bytes|kb|mb|gb)\b'
                ],
                'value_patterns': [
                    r'^\d+(?:\.\d+)?[\s_-]*(?:b|byte|bytes|kb|kib|mb|mib|gb|gib|tb|tib)$',  # With units
                    r'^\d+$'  # Just numbers (assumed bytes)
                ]
            },
            'mime_type': {
                'name_patterns': [
                    r'\b(?:mime|media|content)[\s_-]*(?:type|format)\b',
                    r'\b(?:file|document|image)[\s_-]*(?:mime|media|content)[\s_-]*type\b',
                    r'\b(?:http|internet)[\s_-]*(?:content|media)[\s_-]*type\b'
                ],
                'value_patterns': [
                    r'^[a-z]+/[a-z0-9.+-]+$',  # Standard MIME format
                    r'^(?:text|application|image|audio|video|multipart)/[a-z0-9.+-]+$'  # With common types
                ]
            },
            'hash': {
                'name_patterns': [
                    r'\b(?:hash|checksum|digest|fingerprint|signature)\b',
                    r'\b(?:md5|sha1|sha256|sha512|crc32)[\s_-]*(?:hash|checksum|digest|sum)?\b',
                    r'\b(?:file|document|image|data)[\s_-]*(?:hash|checksum|digest|fingerprint)\b'
                ],
                'value_patterns': [
                    r'^[a-fA-F0-9]{32}$',  # MD5
                    r'^[a-fA-F0-9]{40}$',  # SHA-1
                    r'^[a-fA-F0-9]{64}$'   # SHA-256
                ]
            },
            'version': {
                'name_patterns': [
                    r'\b(?:version|ver|v|release|build|revision)\b',
                    r'\b(?:software|app|application|program|system)[\s_-]*(?:version|ver|release|build)\b',
                    r'\b(?:version|ver|v)[\s_-]*(?:number|id|identifier|string)\b'
                ],
                'value_patterns': [
                    r'^v?\d+(?:\.\d+){0,3}$',  # v1.2.3.4 or 1.2.3.4
                    r'^v?\d+(?:\.\d+){0,3}(?:-[a-zA-Z0-9]+)?$',  # 1.2.3-beta
                    r'^v?\d+(?:\.\d+){0,3}(?:-[a-zA-Z0-9]+)?(?:\+[a-zA-Z0-9]+)?$'  # Semver
                ]
            },
            'timestamp': {
                'name_patterns': [
                    r'\b(?:timestamp|ts|unix[\s_-]*time|epoch|unix[\s_-]*timestamp)\b',
                    r'\b(?:system|server|log|event)[\s_-]*(?:timestamp|time|date)\b',
                    r'\b(?:created|updated|modified|accessed)[\s_-]*(?:timestamp|ts|time)\b'
                ],
                'value_patterns': [
                    r'^\d{10}$',  # Unix timestamp (seconds)
                    r'^\d{13}$'   # Unix timestamp (milliseconds)
                ]
            },
            'log_level': {
                'name_patterns': [
                    r'\b(?:log|logging|logger|severity)[\s_-]*(?:level|priority|severity)\b',
                    r'\b(?:level|priority|severity)[\s_-]*(?:of|for)[\s_-]*(?:log|logging|logger|message)\b',
                    r'\b(?:error|warning|info|debug)[\s_-]*(?:level|priority|severity)\b'
                ],
                'value_patterns': [
                    r'^(?:trace|debug|info|information|notice|warn|warning|error|critical|fatal|emergency)$',
                    r'^(?:verbose|v|d|i|w|e|f)$',  # Shortened versions
                    r'^(?:0|1|2|3|4|5|6|7)$'  # Numeric levels
                ]
            }
        }
    
    def _initialize_international_patterns(self):
        """Initialize international/localization patterns"""
        return {
            'language': {
                'name_patterns': [
                    r'\b(?:language|lang|locale|tongue|speech|dialect)\b',
                    r'\b(?:language|lang)[\s_-]*(?:code|id|identifier|tag|name)\b',
                    r'\b(?:user|customer|visitor|interface|ui|display)[\s_-]*(?:language|lang|locale)\b'
                ],
                'value_patterns': [
                    r'^[a-z]{2}$',  # ISO 639-1 (en, fr, de, etc.)
                    r'^[a-z]{2}-[A-Z]{2}$',  # Language-Country (en-US, fr-FR)
                    r'^(?:english|french|german|spanish|italian|russian|chinese|japanese|korean|arabic)$'  # Common names
                ]
            },
            'country_code': {
                'name_patterns': [
                    r'\b(?:country|nation|land)[\s_-]*(?:code|id|identifier|iso)\b',
                    r'\b(?:iso|international)[\s_-]*(?:country|nation)[\s_-]*code\b',
                    r'\b(?:user|customer|visitor|shipping|billing)[\s_-]*(?:country|nation)[\s_-]*(?:code|id)\b'
                ],
                'value_patterns': [
                    r'^[A-Z]{2}$',  # ISO 3166-1 alpha-2 (US, CA, GB)
                    r'^[A-Z]{3}$'   # ISO 3166-1 alpha-3 (USA, CAN, GBR)
                ]
            },
            'currency_code': {
                'name_patterns': [
                    r'\b(?:currency|money)[\s_-]*(?:code|id|identifier|iso|type)\b',
                    r'\b(?:iso|international)[\s_-]*(?:currency|money)[\s_-]*code\b',
                    r'\b(?:payment|transaction|price|cost)[\s_-]*(?:currency|money)[\s_-]*(?:code|id)\b'
                ],
                'value_patterns': [
                    r'^[A-Z]{3}$'  # ISO 4217 (USD, EUR, GBP)
                ]
            },
            'locale': {
                'name_patterns': [
                    r'\b(?:locale|localization|l10n|internationalization|i18n)\b',
                    r'\b(?:locale|localization)[\s_-]*(?:code|id|identifier|tag|setting)\b',
                    r'\b(?:user|customer|system|interface)[\s_-]*(?:locale|localization|language[\s_-]*and[\s_-]*region)\b'
                ],
                'value_patterns': [
                    r'^[a-z]{2}[-_][A-Z]{2}$',  # Standard locale (en-US, fr-FR)
                    r'^[a-z]{2}[-_][A-Z]{2}\.(?:UTF-8|utf8)$'  # With encoding
                ]
            },
            'timezone': {
                'name_patterns': [
                    r'\b(?:timezone|time[\s_-]*zone|tz|time[\s_-]*offset)\b',
                    r'\b(?:user|customer|system|server|local)[\s_-]*(?:timezone|time[\s_-]*zone|tz)\b',
                    r'\b(?:timezone|time[\s_-]*zone|tz)[\s_-]*(?:id|identifier|name|code|offset)\b'
                ],
                'value_patterns': [
                    r'^(?:GMT|UTC)[+-]\d{1,2}(?::\d{2})?$',  # GMT+8, UTC-05:30
                    r'^[A-Za-z_]+/[A-Za-z_]+$',  # Area/Location (America/New_York)
                    r'^(?:EST|CST|MST|PST|EDT|CDT|MDT|PDT)$'  # Common abbreviations
                ]
            },
            'phone_country_code': {
                'name_patterns': [
                    r'\b(?:phone|telephone|mobile|cell)[\s_-]*(?:country[\s_-]*code|international[\s_-]*code|dialing[\s_-]*code)\b',
                    r'\b(?:international|country|dialing)[\s_-]*(?:phone|telephone|mobile|cell)[\s_-]*code\b',
                    r'\b(?:phone|telephone|mobile|cell)[\s_-]*(?:prefix|int[\s_-]*code)\b'
                ],
                'value_patterns': [
                    r'^\+\d{1,3}$',  # +1, +44
                    r'^\d{1,3}$'     # 1, 44 (without +)
                ]
            },
            'date_format': {
                'name_patterns': [
                    r'\b(?:date|time|datetime)[\s_-]*(?:format|pattern|mask|style)\b',
                    r'\b(?:format|pattern|mask|style)[\s_-]*(?:of|for)[\s_-]*(?:date|time|datetime)\b',
                    r'\b(?:date|time|datetime)[\s_-]*(?:display|presentation|representation)\b'
                ],
                'value_patterns': [
                    r'^(?:MM/dd/yyyy|dd/MM/yyyy|yyyy-MM-dd|yyyy/MM/dd|M/d/yyyy|d/M/yyyy)$',  # Date formats
                    r'^(?:HH:mm:ss|hh:mm:ss a|HH:mm|h:mm a)$',  # Time formats
                    r'^(?:yyyy-MM-dd HH:mm:ss|MM/dd/yyyy hh:mm:ss a)$'  # Combined formats
                ]
            },
            'number_format': {
                'name_patterns': [
                    r'\b(?:number|numeric|decimal|float|integer)[\s_-]*(?:format|pattern|mask|style)\b',
                    r'\b(?:format|pattern|mask|style)[\s_-]*(?:of|for)[\s_-]*(?:number|numeric|decimal|float|integer)\b',
                    r'\b(?:number|numeric|decimal|float|integer)[\s_-]*(?:display|presentation|representation)\b'
                ],
                'value_patterns': [
                    r'^(?:#,##0|#,##0.00|#,##0.###|0.###|0.00)$',  # Format patterns
                    r'^(?:1,234|1,234.56|1 234,56|1.234,56)$'  # Examples of formatted numbers
                ]
            }
        }
        
    def _analyze_dtype(self, series):
        """
        Analyze the data type of a series and determine the base type
        
        Args:
            series (pandas.Series): The column data to analyze
            
        Returns:
            dict: Base type information and dtype-specific details
        """
        result = {
            'base_type': 'unknown',
            'details': {}
        }
        
        # Drop NA values for better type inference
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return result
        
        dtype_name = str(series.dtype)
        
        # Check for numeric types
        if pd.api.types.is_numeric_dtype(series):
            result['base_type'] = 'numeric'
            if pd.api.types.is_integer_dtype(series):
                result['details']['numeric_type'] = 'integer'
            elif pd.api.types.is_float_dtype(series):
                result['details']['numeric_type'] = 'float'
            else:
                result['details']['numeric_type'] = 'other'
        
        # Check for datetime types
        elif pd.api.types.is_datetime64_dtype(series):
            result['base_type'] = 'datetime'
            result['details']['has_time'] = (clean_series.dt.hour != 0).any() or (clean_series.dt.minute != 0).any()
        
        # Check for boolean types
        elif pd.api.types.is_bool_dtype(series):
            result['base_type'] = 'boolean'
        
        # Handle string and object types
        elif dtype_name in ['object', 'string']:
            # Sample to check if it's a string type or mixed
            sample = clean_series.sample(min(len(clean_series), self.max_sample_size))
            
            # Check if all elements are strings
            if all(isinstance(x, str) for x in sample):
                result['base_type'] = 'string'
                
                # Try to detect if strings represent dates or numbers
                try:
                    # Check if strings can be parsed as dates
                    pd.to_datetime(sample, errors='raise')
                    result['base_type'] = 'datetime'
                    result['details']['format'] = 'string_date'
                except:
                    # Check if strings can be parsed as numbers
                    if all(self._is_numeric_string(x) for x in sample):
                        result['base_type'] = 'numeric'
                        result['details']['numeric_type'] = 'string_numeric'
                        
                        # Determine if all strings represent integers
                        if all(float(x).is_integer() for x in sample if self._is_numeric_string(x)):
                            result['details']['numeric_type'] = 'string_integer'
            
        return result
    
    def _is_numeric_string(self, value):
        """
        Check if a string represents a numeric value
        
        Args:
            value (str): The string to check
            
        Returns:
            bool: True if the string can be converted to a number
        """
        if not isinstance(value, str):
            return False
        
        # Remove spaces, commas, currency symbols
        cleaned = value.strip().replace(',', '').replace('$', '').replace('€', '').replace('£', '')
        
        # Check if it can be converted to float
        try:
            float(cleaned)
            return True
        except ValueError:
            return False
        
        return False
    
    def _analyze_numeric_column(self, original_series, clean_series, dtype_info):
        """
        Analyze a numeric column to identify its specific type
        
        Args:
            original_series (pandas.Series): The original column data
            clean_series (pandas.Series): The cleaned column data (no NAs)
            dtype_info (dict): Information about the column's dtype
            
        Returns:
            dict: Analysis results including type, confidence, and statistics
        """
        result = {
            'type': 'numeric',
            'confidence': 0.7,  # Default confidence for numeric type
            'subtypes': {},
            'statistics': self._get_numeric_statistics(original_series, clean_series)
        }
        
        # Determine more specific numeric types based on name and values
        column_name = original_series.name
        column_name_lower = str(column_name).lower() if column_name is not None else ""
        
        # Check for common patterns in column name
        pattern_matches = {}
        for pattern_type in ['product_id', 'customer_id', 'order_id', 'price', 'quantity', 'revenue', 'sales', 'discount', 'tax']:
            # Get the pattern dictionary based on the pattern type
            pattern_dict = self.all_patterns.get(pattern_type, {})
            if not pattern_dict:
                continue
            
            # Check name patterns
            name_patterns = pattern_dict.get('name_patterns', [])
            for pattern in name_patterns:
                if re.search(pattern, column_name_lower):
                    pattern_matches[pattern_type] = pattern_matches.get(pattern_type, 0) + 1
        
        # Determine the most likely pattern match
        if pattern_matches:
            best_match = max(pattern_matches.items(), key=lambda x: x[1])
            result['type'] = best_match[0]
            result['confidence'] = 0.5 + (0.1 * best_match[1])  # Increase confidence based on number of matches
        
        # For integer columns, check if they might be IDs
        if dtype_info.get('details', {}).get('numeric_type') in ['integer', 'string_integer']:
            if any(re.search(pattern, column_name_lower) for pattern in self.all_patterns.get('id', {}).get('name_patterns', [])):
                result['type'] = 'id'
                result['confidence'] = 0.8
        
        return result
    
    def _analyze_string_column(self, original_series, clean_series, dtype_info):
        """
        Analyze a string column to identify its specific type
        
        Args:
            original_series (pandas.Series): The original column data
            clean_series (pandas.Series): The cleaned column data (no NAs)
            dtype_info (dict): Information about the column's dtype
            
        Returns:
            dict: Analysis results including type, confidence, and statistics
        """
        result = {
            'type': 'string',
            'confidence': 0.6,  # Default confidence for string type
            'subtypes': {},
            'statistics': self._get_string_statistics(original_series, clean_series)
        }
        
        # Determine more specific string types based on name and values
        column_name = original_series.name
        column_name_lower = str(column_name).lower() if column_name is not None else ""
        
        # Sample values for pattern matching
        sample_values = clean_series.sample(min(len(clean_series), self.max_sample_size))
        sample_values_lower = [str(v).lower() for v in sample_values if isinstance(v, str)]
        
        # Check for common patterns in column name and values
        pattern_matches = {}
        
        # Prioritize checking for specific types
        for check_type in ['email', 'url', 'phone', 'ip_address', 'date', 'name', 'address', 'city', 'state_province', 
                            'postal_code', 'country', 'product_name', 'customer_name', 'product_category', 'status']:
            
            # Get the pattern dictionary based on the check type
            pattern_dict = {}
            for pattern_prefix in ['', 'core_', 'product_', 'customer_', 'location_']:
                pattern_key = f"{pattern_prefix}{check_type}"
                if pattern_key in self.all_patterns:
                    pattern_dict = self.all_patterns[pattern_key]
                    break
                    
            if not pattern_dict:
                continue
            
            # Check name patterns
            name_match = False
            name_patterns = pattern_dict.get('name_patterns', [])
            for pattern in name_patterns:
                if re.search(pattern, column_name_lower):
                    pattern_matches[check_type] = pattern_matches.get(check_type, 0) + 2  # Higher weight for name match
                    name_match = True
                    break
            
            # Check value patterns if name matched or for certain important types
            if name_match or check_type in ['email', 'url', 'phone', 'ip_address', 'date']:
                value_patterns = pattern_dict.get('value_patterns', [])
                # Handle both list and dict patterns
                if isinstance(value_patterns, dict):
                    value_patterns = [v for v in value_patterns.values()]
                
                value_match_count = 0
                for sample_value in sample_values_lower:
                    for pattern in value_patterns:
                        if re.search(pattern, sample_value):
                            value_match_count += 1
                            break
                
                # Calculate percentage of matching values
                if value_match_count > 0:
                    match_percentage = value_match_count / len(sample_values)
                    if match_percentage > 0.5:  # More than half match
                        pattern_matches[check_type] = pattern_matches.get(check_type, 0) + (match_percentage * 3)  # Weight by percentage
        
        # Determine the most likely pattern match
        if pattern_matches:
            best_match = max(pattern_matches.items(), key=lambda x: x[1])
            result['type'] = best_match[0]
            result['confidence'] = min(0.5 + (0.1 * best_match[1]), 0.95)  # Increase confidence based on match strength
        
        return result
    
    def _analyze_datetime_column(self, original_series, clean_series, dtype_info):
        """
        Analyze a datetime column to identify its specific type
        
        Args:
            original_series (pandas.Series): The original column data
            clean_series (pandas.Series): The cleaned column data (no NAs)
            dtype_info (dict): Information about the column's dtype
            
        Returns:
            dict: Analysis results including type, confidence, and statistics
        """
        result = {
            'type': 'date',
            'confidence': 0.8,  # Default confidence for date type
            'subtypes': {},
            'statistics': self._get_datetime_statistics(original_series, clean_series)
        }
        
        # Determine more specific datetime types based on name
        column_name = original_series.name
        column_name_lower = str(column_name).lower() if column_name is not None else ""
        
        # Check for common datetime patterns in column name
        datetime_types = {
            'order_date': ['order.*date', 'purchase.*date', 'transaction.*date'],
            'delivery_date': ['delivery.*date', 'shipping.*date', 'arrival.*date'],
            'created_date': ['create.*date', 'creation.*date', 'added.*date'],
            'modified_date': ['modified.*date', 'update.*date', 'change.*date'],
            'birth_date': ['birth.*date', 'dob', 'born.*date'],
            'expiry_date': ['expiry.*date', 'expiration.*date', 'expire.*date']
        }
        
        for date_type, patterns in datetime_types.items():
            for pattern in patterns:
                if re.search(pattern, column_name_lower):
                    result['type'] = date_type
                    result['confidence'] = 0.9
                    break
        
        # Check if it has time component
        has_time = dtype_info.get('details', {}).get('has_time', False)
        if has_time:
            if result['type'] == 'date':
                result['type'] = 'datetime'
            else:
                result['subtypes']['has_time'] = True
        
        return result
    
    def _analyze_boolean_column(self, original_series, clean_series, dtype_info):
        """
        Analyze a boolean column to identify its specific type
        
        Args:
            original_series (pandas.Series): The original column data
            clean_series (pandas.Series): The cleaned column data (no NAs)
            dtype_info (dict): Information about the column's dtype
            
        Returns:
            dict: Analysis results including type, confidence, and statistics
        """
        result = {
            'type': 'boolean',
            'confidence': 0.9,  # High confidence for boolean type
            'subtypes': {},
            'statistics': self._get_boolean_statistics(original_series, clean_series)
        }
        
        # Determine more specific boolean types based on name
        column_name = original_series.name
        column_name_lower = str(column_name).lower() if column_name is not None else ""
        
        # Check for common boolean flag patterns in column name
        boolean_types = {
            'active_flag': ['is.*active', 'active.*flag', 'status.*active'],
            'approved_flag': ['is.*approved', 'approved.*flag', 'status.*approved'],
            'deleted_flag': ['is.*deleted', 'deleted.*flag', 'status.*deleted'],
            'available_flag': ['is.*available', 'available.*flag', 'in.*stock'],
            'completed_flag': ['is.*completed', 'completed.*flag', 'status.*completed']
        }
        
        for flag_type, patterns in boolean_types.items():
            for pattern in patterns:
                if re.search(pattern, column_name_lower):
                    result['type'] = flag_type
                    result['confidence'] = 0.95
                    break
        
        return result
    
    def _get_numeric_statistics(self, original_series, clean_series):
        """
        Calculate statistics for numeric columns
        
        Args:
            original_series (pandas.Series): The original column data
            clean_series (pandas.Series): The cleaned column data (no NAs)
            
        Returns:
            dict: Statistics about the numeric column
        """
        stats = {
            'count': len(clean_series),
            'nulls': len(original_series) - len(clean_series),
            'null_percentage': (len(original_series) - len(clean_series)) / len(original_series) if len(original_series) > 0 else 0,
            'unique_count': clean_series.nunique(),
            'unique_percentage': clean_series.nunique() / len(clean_series) if len(clean_series) > 0 else 0
        }
        
        # Add basic descriptive statistics if there's data
        if len(clean_series) > 0:
            stats.update({
                'min': float(clean_series.min()),
                'max': float(clean_series.max()),
                'mean': float(clean_series.mean()),
                'median': float(clean_series.median()),
                'std': float(clean_series.std()) if len(clean_series) > 1 else 0,
                'zeros': int((clean_series == 0).sum()),
                'zeros_percentage': float((clean_series == 0).sum() / len(clean_series)) if len(clean_series) > 0 else 0,
                'negatives': int((clean_series < 0).sum()),
                'negatives_percentage': float((clean_series < 0).sum() / len(clean_series)) if len(clean_series) > 0 else 0
            })
        
        return stats
    
    def _get_string_statistics(self, original_series, clean_series):
        """
        Calculate statistics for string columns
        
        Args:
            original_series (pandas.Series): The original column data
            clean_series (pandas.Series): The cleaned column data (no NAs)
            
        Returns:
            dict: Statistics about the string column
        """
        stats = {
            'count': len(clean_series),
            'nulls': len(original_series) - len(clean_series),
            'null_percentage': (len(original_series) - len(clean_series)) / len(original_series) if len(original_series) > 0 else 0,
            'unique_count': clean_series.nunique(),
            'unique_percentage': clean_series.nunique() / len(clean_series) if len(clean_series) > 0 else 0
        }
        
        # Add string-specific statistics if there's data
        if len(clean_series) > 0:
            # Convert to strings and compute length
            str_series = clean_series.astype(str)
            length_series = str_series.str.len()
            
            stats.update({
                'min_length': int(length_series.min()),
                'max_length': int(length_series.max()),
                'mean_length': float(length_series.mean()),
                'empty_strings': int((str_series == '').sum()),
                'empty_percentage': float((str_series == '').sum() / len(clean_series)) if len(clean_series) > 0 else 0
            })
            
            # Calculate most common values (top 5)
            top_values = clean_series.value_counts().head(5).to_dict()
            stats['top_values'] = {str(k): int(v) for k, v in top_values.items()}
        
        return stats
    
    def _get_datetime_statistics(self, original_series, clean_series):
        """
        Calculate statistics for datetime columns
        
        Args:
            original_series (pandas.Series): The original column data
            clean_series (pandas.Series): The cleaned column data (no NAs)
            
        Returns:
            dict: Statistics about the datetime column
        """
        stats = {
            'count': len(clean_series),
            'nulls': len(original_series) - len(clean_series),
            'null_percentage': (len(original_series) - len(clean_series)) / len(original_series) if len(original_series) > 0 else 0,
            'unique_count': clean_series.nunique(),
            'unique_percentage': clean_series.nunique() / len(clean_series) if len(clean_series) > 0 else 0
        }
        
        # Add datetime-specific statistics if there's data
        if len(clean_series) > 0:
            try:
                stats.update({
                    'min_date': clean_series.min().strftime('%Y-%m-%d %H:%M:%S'),
                    'max_date': clean_series.max().strftime('%Y-%m-%d %H:%M:%S'),
                    'date_range_days': (clean_series.max() - clean_series.min()).days,
                    'has_time': (clean_series.dt.hour != 0).any() or (clean_series.dt.minute != 0).any(),
                    'future_dates': int((clean_series > pd.Timestamp.now()).sum()),
                    'future_percentage': float((clean_series > pd.Timestamp.now()).sum() / len(clean_series)) if len(clean_series) > 0 else 0
                })
            except (AttributeError, TypeError):
                # Handle case where datetime operations fail
                stats.update({
                    'min_date': str(clean_series.min()),
                    'max_date': str(clean_series.max()),
                    'date_range_days': None,
                    'has_time': None,
                    'future_dates': None,
                    'future_percentage': None
                })
        
        return stats
    
    def _get_boolean_statistics(self, original_series, clean_series):
        """
        Calculate statistics for boolean columns
        
        Args:
            original_series (pandas.Series): The original column data
            clean_series (pandas.Series): The cleaned column data (no NAs)
            
        Returns:
            dict: Statistics about the boolean column
        """
        stats = {
            'count': len(clean_series),
            'nulls': len(original_series) - len(clean_series),
            'null_percentage': (len(original_series) - len(clean_series)) / len(original_series) if len(original_series) > 0 else 0
        }
        
        # Add boolean-specific statistics if there's data
        if len(clean_series) > 0:
            true_count = clean_series.sum()
            stats.update({
                'true_count': int(true_count),
                'false_count': int(len(clean_series) - true_count),
                'true_percentage': float(true_count / len(clean_series)) if len(clean_series) > 0 else 0,
                'false_percentage': float((len(clean_series) - true_count) / len(clean_series)) if len(clean_series) > 0 else 0
            })
        
        return stats
    
    def analyze_dataframe(self, df, **kwargs):
        """
        Analyze an entire dataframe to identify column types
        
        Args:
            df (pandas.DataFrame): The dataframe to analyze
            **kwargs: Additional arguments
            
        Returns:
            dict: Analysis results for all columns
        """
        results = {}
        
        # Process each column
        for column_name in df.columns:
            series = df[column_name]
            results[column_name] = self.analyze_column(series)
            
        return results