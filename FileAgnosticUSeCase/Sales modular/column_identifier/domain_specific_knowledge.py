"""
Domain-Specific Knowledge Models for Column Identification

This module provides domain-specific patterns and vocabularies for various industries
to enhance column identification accuracy in specialized datasets.
"""
import re
class DomainKnowledgeBase:
    """
    Domain-specific knowledge base containing specialized patterns and vocabularies
    for various industries to enhance column identification accuracy.
    """
    
    def __init__(self):
        """Initialize the domain knowledge base with industry-specific patterns"""
        self.domain_patterns = self._initialize_domain_patterns()
        self.domain_vocabularies = self._initialize_domain_vocabularies()
        self.domain_abbreviations = self._initialize_domain_abbreviations()
        
    def _initialize_domain_patterns(self):
        """Initialize domain-specific patterns for column identification"""
        return {
            # Retail industry patterns
            'retail': {
                'sales': [
                    r'\b(?:pos|register|checkout)[\s_-]*(?:amount|total|value|sale|revenue)\b',
                    r'\b(?:basket|cart|receipt|ticket)[\s_-]*(?:total|value|amount)\b',
                    r'\b(?:trx|tran|transaction)[\s_-]*(?:amount|total|value|sale)\b',
                    r'\b(?:net|gross)[\s_-]*(?:sales|receipts|takings)\b'
                ],
                'inventory': [
                    r'\b(?:inventory|stock|on[\s_-]*hand|oh|on[\s_-]*shelf)[\s_-]*(?:quantity|qty|count|level|units|amount)\b',
                    r'\b(?:beginning|ending|closing|opening)[\s_-]*(?:inventory|stock)\b',
                    r'\b(?:stockout|out[\s_-]*of[\s_-]*stock|oos)[\s_-]*(?:status|flag|indicator)?\b',
                    r'\b(?:safety|buffer|min|minimum|max|maximum|reorder)[\s_-]*(?:stock|level|point|qty|quantity)\b'
                ],
                'product': [
                    r'\b(?:upc|ean|isbn|gtin|sku|style[\s_-]*number|model[\s_-]*number)\b',
                    r'\b(?:assortment|collection|range|line|brand|vendor|supplier)[\s_-]*(?:code|id|name)?\b',
                    r'\b(?:department|category|class|family|group)[\s_-]*(?:code|id|name|number)?\b',
                    r'\b(?:size|color|finish|flavor|variant|attribute|option)[\s_-]*(?:code|id|name|value)?\b'
                ],
                'pricing': [
                    r'\b(?:retail|list|sale|promo|regular|base|current|future|everyday|competitive)[\s_-]*price\b',
                    r'\b(?:msrp|map|srp)[\s_-]*(?:price)?\b',
                    r'\b(?:cost|vendor|purchase|acquisition)[\s_-]*price\b',
                    r'\b(?:markdown|discount|promo|promotion|reduction)[\s_-]*(?:amount|percent|pct|rate|value)?\b',
                    r'\b(?:margin|markup|profit)[\s_-]*(?:percent|pct|rate|amount|dollar|value)?\b'
                ],
                'store': [
                    r'\b(?:store|shop|outlet|location|branch|site)[\s_-]*(?:id|number|code|name|no)?\b',
                    r'\b(?:store|location)[\s_-]*(?:type|format|size|tier|grade|class)\b',
                    r'\b(?:district|region|zone|area|territory|market)[\s_-]*(?:id|number|code|name|no)?\b',
                    r'\b(?:trade[\s_-]*area|catchment|dma|msa)[\s_-]*(?:id|code|name)?\b'
                ]
            },
            
            # Healthcare industry patterns
            'healthcare': {
                'patient': [
                    r'\b(?:patient|pt|client|member|beneficiary)[\s_-]*(?:id|number|code|identifier|no|mrn)\b',
                    r'\b(?:emr|ehr|chart|record|account)[\s_-]*(?:id|number|no)\b',
                    r'\b(?:ssn|social[\s_-]*security)[\s_-]*(?:number|no)?\b',
                    r'\b(?:dob|birth[\s_-]*date|birthdate)\b',
                    r'\b(?:age|years[\s_-]*old)\b'
                ],
                'provider': [
                    r'\b(?:provider|physician|doctor|clinician|practitioner)[\s_-]*(?:id|number|code|npi|identifier|no)\b',
                    r'\b(?:specialty|specialization|department|service[\s_-]*line)[\s_-]*(?:code|id|name)?\b',
                    r'\b(?:facility|hospital|clinic|location|practice)[\s_-]*(?:id|code|number|name|no)?\b',
                    r'\b(?:badge|credential|license|dea)[\s_-]*(?:id|number|no)?\b'
                ],
                'clinical': [
                    r'\b(?:diagnosis|dx|condition|problem|complaint)[\s_-]*(?:code|id|name)?\b',
                    r'\b(?:icd|icd-?9|icd-?10|snomed|dsm)[\s_-]*(?:code|id|number)?\b',
                    r'\b(?:procedure|proc|treatment|intervention|surgery|service)[\s_-]*(?:code|id|name)?\b',
                    r'\b(?:cpt|hcpcs|drg)[\s_-]*(?:code|id|number)?\b',
                    r'\b(?:medication|drug|rx|prescription|med)[\s_-]*(?:code|id|name|ndc)?\b'
                ],
                'claims': [
                    r'\b(?:claim|bill|invoice|encounter)[\s_-]*(?:id|number|code|no)\b',
                    r'\b(?:charge|fee|payment|reimbursement|adjustment)[\s_-]*(?:amount|code|id)?\b',
                    r'\b(?:allowable|allowed|approved|covered|eligible)[\s_-]*(?:amount|charge|payment)\b',
                    r'\b(?:copay|coinsurance|deductible|oop|out[\s_-]*of[\s_-]*pocket)[\s_-]*(?:amount)?\b',
                    r'\b(?:denial|rejection)[\s_-]*(?:code|reason|flag|indicator)?\b'
                ],
                'insurance': [
                    r'\b(?:insurance|payer|carrier|plan|program)[\s_-]*(?:id|code|name|number|no)?\b',
                    r'\b(?:member|subscriber|insured|beneficiary|enrollee)[\s_-]*(?:id|number|code|no)?\b',
                    r'\b(?:group|policy|contract|coverage)[\s_-]*(?:id|number|code|no)?\b',
                    r'\b(?:authorization|auth|precert|approval|referral)[\s_-]*(?:id|number|code|no)?\b',
                    r'\b(?:eligibility|enrollment|coverage)[\s_-]*(?:status|indicator|flag|start|end|date)?\b'
                ]
            },
            
            # Financial industry patterns
            'finance': {
                'account': [
                    r'\b(?:account|acct|acn)[\s_-]*(?:id|number|code|no)\b',
                    r'\b(?:account|acct)[\s_-]*(?:type|status|tier|level|class|category)\b',
                    r'\b(?:checking|savings|deposit|cd|money[\s_-]*market|custody|investment|retirement)[\s_-]*account\b',
                    r'\b(?:folio|portfolio|fund|holding)[\s_-]*(?:id|number|code|no)?\b',
                    r'\b(?:wallet|purse|user[\s_-]*id|member[\s_-]*id|customer[\s_-]*id)[\s_-]*(?:id|number|code|no)?\b'
                ],
                'transaction': [
                    r'\b(?:transaction|txn|tran)[\s_-]*(?:id|number|code|reference|ref|no)\b',
                    r'\b(?:transaction|txn|tran)[\s_-]*(?:type|category|description|desc|memo)\b',
                    r'\b(?:posting|settle|settlement|effective|value)[\s_-]*date\b',
                    r'\b(?:debit|credit|withdrawal|deposit|transfer|payment)[\s_-]*(?:amount|value|date)?\b',
                    r'\b(?:entry|ledger|journal|gl)[\s_-]*(?:id|number|code|no)?\b'
                ],
                'instrument': [
                    r'\b(?:instrument|security|asset|position|holding)[\s_-]*(?:id|code|ticker|symbol|cusip|isin|sedol)\b',
                    r'\b(?:stock|equity|bond|fixed[\s_-]*income|option|future|derivative|etf|mutual[\s_-]*fund)[\s_-]*(?:id|code|symbol)?\b',
                    r'\b(?:currency|fx|forex)[\s_-]*(?:code|type|pair)?\b',
                    r'\b(?:price|nav|quote|bid|ask|spread|yield|rate|ratio)[\s_-]*(?:value|amount)?\b',
                    r'\b(?:shares|units|quantity|position|notional|face[\s_-]*value|principal)[\s_-]*(?:amount|value|count)?\b'
                ],
                'risk': [
                    r'\b(?:risk|volatility|var|value[\s_-]*at[\s_-]*risk)[\s_-]*(?:score|rating|grade|measure|level|profile|exposure)\b',
                    r'\b(?:credit|counterparty|market|operational|liquidity)[\s_-]*risk\b',
                    r'\b(?:limit|cap|threshold|exposure|concentration|utilization)[\s_-]*(?:value|amount|percent|rate)?\b',
                    r'\b(?:compliance|restriction|regulation|rule|policy)[\s_-]*(?:status|flag|indicator)?\b',
                    r'\b(?:rating|grade|score|tier|rank|classification)[\s_-]*(?:value|code|level)?\b'
                ],
                'performance': [
                    r'\b(?:return|performance|yield|gain|loss|profit)[\s_-]*(?:amount|value|percent|pct|rate|ratio)?\b',
                    r'\b(?:absolute|relative|excess|alpha|beta|active)[\s_-]*return\b',
                    r'\b(?:ytd|mtd|qtd|1yr|3yr|5yr|since[\s_-]*inception)[\s_-]*(?:return|performance)\b',
                    r'\b(?:sharpe|sortino|information|treynor)[\s_-]*ratio\b',
                    r'\b(?:benchmark|index|hurdle|target)[\s_-]*(?:return|value|performance)?\b'
                ]
            },
            
            # Manufacturing industry patterns
            'manufacturing': {
                'product': [
                    r'\b(?:product|item|part|component|assembly|material|raw[\s_-]*material)[\s_-]*(?:id|number|code|no)\b',
                    r'\b(?:bom|bill[\s_-]*of[\s_-]*materials)[\s_-]*(?:id|number|code|no)?\b',
                    r'\b(?:make|model|style|version|variant|revision|rev)[\s_-]*(?:id|number|code|no)?\b',
                    r'\b(?:specification|spec|tolerance|dimension|parameter)[\s_-]*(?:id|code|value|name)?\b',
                    r'\b(?:uom|unit[\s_-]*of[\s_-]*measure)[\s_-]*(?:id|code)?\b'
                ],
                'production': [
                    r'\b(?:production|manufacturing|assembly|work)[\s_-]*order\b',
                    r'\b(?:job|batch|lot|run|series|campaign)[\s_-]*(?:id|number|code|no)?\b',
                    r'\b(?:process|operation|step|stage|phase|activity)[\s_-]*(?:id|number|code|name|no)?\b',
                    r'\b(?:cycle|takt|lead|processing|setup|idle|downtime)[\s_-]*time\b',
                    r'\b(?:start|end|scheduled|actual|planned|estimated)[\s_-]*(?:time|date)\b'
                ],
                'equipment': [
                    r'\b(?:equipment|machine|tool|device|asset|resource)[\s_-]*(?:id|number|code|name|no)?\b',
                    r'\b(?:capacity|capability|utilization|efficiency|oee|availability)[\s_-]*(?:rate|value|percent|factor)?\b',
                    r'\b(?:maintenance|calibration|service|repair)[\s_-]*(?:date|schedule|status|type)?\b',
                    r'\b(?:station|cell|line|facility|plant|site|location)[\s_-]*(?:id|number|code|name|no)?\b',
                    r'\b(?:shift|crew|team|operator|supervisor)[\s_-]*(?:id|number|code|name|no)?\b'
                ],
                'quality': [
                    r'\b(?:quality|qc|qa|inspection)[\s_-]*(?:id|number|code|result|status|check)\b',
                    r'\b(?:defect|nonconformance|nc|deviation|variance|reject)[\s_-]*(?:id|code|count|type|rate|percent)?\b',
                    r'\b(?:test|check|measurement|sample|audit)[\s_-]*(?:id|number|result|value|data)?\b',
                    r'\b(?:pass|fail|accept|reject|hold|release)[\s_-]*(?:status|flag|indicator|result)?\b',
                    r'\b(?:rework|scrap|waste|yield|first[\s_-]*pass[\s_-]*yield|fpy)[\s_-]*(?:count|amount|rate|percent)?\b'
                ],
                'supply_chain': [
                    r'\b(?:vendor|supplier|manufacturer|source|provider)[\s_-]*(?:id|number|code|name|no)?\b',
                    r'\b(?:purchase|po|procurement|requisition|material)[\s_-]*order\b',
                    r'\b(?:receipt|delivery|shipment|transfer|movement)[\s_-]*(?:id|number|code|no)?\b',
                    r'\b(?:lead|delivery|transit|cycle|fulfillment|replenishment)[\s_-]*time\b',
                    r'\b(?:warehouse|dc|distribution[\s_-]*center|facility|location|bin|rack|aisle|zone)[\s_-]*(?:id|code|number|no)?\b'
                ]
            },
            
            # E-commerce industry patterns
            'ecommerce': {
                'order': [
                    r'\b(?:order|checkout|purchase|transaction)[\s_-]*(?:id|number|code|no|reference|ref)\b',
                    r'\b(?:order|purchase)[\s_-]*(?:date|time|datetime|timestamp)\b',
                    r'\b(?:order|purchase)[\s_-]*(?:status|state|condition|phase)\b',
                    r'\b(?:cart|basket|bag)[\s_-]*(?:id|total|value|size|items|count)\b',
                    r'\b(?:line[\s_-]*item|order[\s_-]*line|order[\s_-]*detail)[\s_-]*(?:id|number|code|no)?\b'
                ],
                'product': [
                    r'\b(?:product|item|listing|offer|catalog[\s_-]*entry)[\s_-]*(?:id|number|code|sku|pid|no)\b',
                    r'\b(?:product|listing)[\s_-]*(?:title|name|headline|label)\b',
                    r'\b(?:brand|manufacturer|vendor|seller|merchant)[\s_-]*(?:id|name|code)?\b',
                    r'\b(?:condition|state|quality)[\s_-]*(?:code|value|level|grade|rating)?\b',
                    r'\b(?:variant|option|attribute|feature|specification|color|size)[\s_-]*(?:id|name|value|code)?\b'
                ],
                'customer': [
                    r'\b(?:customer|user|buyer|shopper|account)[\s_-]*(?:id|number|code|no)\b',
                    r'\b(?:guest|anonymous|registered|logged[\s_-]*in|new|returning)[\s_-]*(?:user|customer|visitor)\b',
                    r'\b(?:login|username|user[\s_-]*id|email|handle|account[\s_-]*id)[\s_-]*(?:value)?\b',
                    r'\b(?:loyalty|rewards|points|tier|level|status|vip)[\s_-]*(?:id|value|amount|balance)?\b',
                    r'\b(?:wishlist|favorites|saved[\s_-]*items|cart)[\s_-]*(?:id|count|size)?\b'
                ],
                'payment': [
                    r'\b(?:payment|transaction|charge)[\s_-]*(?:id|number|code|reference|ref|no)\b',
                    r'\b(?:payment|transaction)[\s_-]*(?:status|state|result|outcome)\b',
                    r'\b(?:payment|payment[\s_-]*method|payment[\s_-]*type)[\s_-]*(?:id|code|name)?\b',
                    r'\b(?:credit[\s_-]*card|debit[\s_-]*card|bank[\s_-]*account|wallet|paypal|apple[\s_-]*pay|google[\s_-]*pay)[\s_-]*(?:id|number|last4)?\b',
                    r'\b(?:authorization|settlement|capture|void|refund|chargeback)[\s_-]*(?:id|code|status|amount)?\b'
                ],
                'shipping': [
                    r'\b(?:shipping|delivery|fulfillment|shipment)[\s_-]*(?:id|number|code|reference|ref|no)\b',
                    r'\b(?:shipping|delivery|fulfillment)[\s_-]*(?:method|type|option|service|carrier)\b',
                    r'\b(?:shipping|delivery|fulfillment|shipment)[\s_-]*(?:cost|fee|charge|price|amount)\b',
                    r'\b(?:tracking|shipment|package|parcel)[\s_-]*(?:id|number|code|no)\b',
                    r'\b(?:shipping|delivery|arrival|expected|estimated)[\s_-]*(?:date|time|window)\b'
                ]
            },
            
            # Tech/SaaS industry patterns
            'tech_saas': {
                'user': [
                    r'\b(?:user|account|profile|member|subscriber)[\s_-]*(?:id|number|code|no)\b',
                    r'\b(?:username|login|handle|email|uid)[\s_-]*(?:id|value)?\b',
                    r'\b(?:user|account|subscription)[\s_-]*(?:type|status|state|tier|level|role|plan)\b',
                    r'\b(?:authentication|auth|security|access|permission|privilege)[\s_-]*(?:level|role|group|type)?\b',
                    r'\b(?:signup|registration|join|onboarding|activation)[\s_-]*(?:date|time|source|channel)?\b'
                ],
                'subscription': [
                    r'\b(?:subscription|plan|package|tier|membership)[\s_-]*(?:id|code|name|type|level)?\b',
                    r'\b(?:subscription|plan|membership)[\s_-]*(?:start|end|renewal|expiry|expiration)[\s_-]*(?:date)?\b',
                    r'\b(?:billing|payment|invoice|charge)[\s_-]*(?:cycle|period|frequency|interval|date)\b',
                    r'\b(?:trial|freemium|premium|pro|enterprise|business)[\s_-]*(?:plan|subscription|tier|level)?\b',
                    r'\b(?:auto[\s_-]*renew|recurring|renewal|cancellation|churn|retention)[\s_-]*(?:status|flag|date|rate)?\b'
                ],
                'usage': [
                    r'\b(?:usage|consumption|utilization)[\s_-]*(?:amount|value|count|limit|quota|rate)\b',
                    r'\b(?:api|call|request|query|transaction)[\s_-]*(?:count|limit|usage|rate|volume)\b',
                    r'\b(?:storage|bandwidth|throughput|compute|processing)[\s_-]*(?:usage|amount|limit|quota)?\b',
                    r'\b(?:active|concurrent|simultaneous|peak|average)[\s_-]*(?:users|sessions|connections|licenses)\b',
                    r'\b(?:time[\s_-]*spent|duration|engagement|session[\s_-]*length|activity)[\s_-]*(?:value|amount|time)?\b'
                ],
                'event': [
                    r'\b(?:event|action|activity|behavior|interaction)[\s_-]*(?:id|type|name|category|code)?\b',
                    r'\b(?:event|action|log|audit)[\s_-]*(?:timestamp|time|date|datetime)\b',
                    r'\b(?:click|view|impression|pageview|visit|session)[\s_-]*(?:id|count|time|duration)?\b',
                    r'\b(?:conversion|acquisition|activation|retention|referral|revenue)[\s_-]*(?:event|action|trigger)?\b',
                    r'\b(?:source|medium|campaign|channel|referrer|utm)[\s_-]*(?:id|name|type|value)?\b'
                ],
                'performance': [
                    r'\b(?:response|load|processing|execution|latency|request)[\s_-]*time\b',
                    r'\b(?:error|exception|crash|bug|incident|failure)[\s_-]*(?:id|code|type|count|rate|percent)?\b',
                    r'\b(?:availability|uptime|downtime|sla|reliability)[\s_-]*(?:percent|rate|status|score)?\b',
                    r'\b(?:cpu|memory|disk|storage|bandwidth|throughput)[\s_-]*(?:usage|utilization|consumption)?\b',
                    r'\b(?:resource|server|instance|node|pod|container|cluster)[\s_-]*(?:id|count|capacity|size)?\b'
                ]
            }
        }
        
    def _initialize_domain_vocabularies(self):
        """Initialize domain-specific vocabularies for enhanced identification"""
        return {
            'retail': {
                'product_types': [
                    'apparel', 'clothing', 'footwear', 'accessory', 'jewelry', 'electronics', 
                    'appliance', 'furniture', 'houseware', 'grocery', 'produce', 'dairy', 
                    'bakery', 'meat', 'seafood', 'frozen', 'beverage', 'alcohol', 'health', 
                    'beauty', 'pharmacy', 'toy', 'sporting', 'outdoor', 'garden', 'automotive',
                    'hardware', 'tool', 'office', 'book', 'music', 'video', 'gift', 'seasonal'
                ],
                'store_types': [
                    'supermarket', 'hypermarket', 'department', 'specialty', 'convenience', 
                    'discount', 'warehouse', 'club', 'outlet', 'flagship', 'popup', 'kiosk',
                    'mall', 'strip', 'center', 'standalone', 'downtown', 'suburban', 'rural',
                    'concept', 'showroom'
                ],
                'transaction_types': [
                    'sale', 'return', 'exchange', 'refund', 'void', 'adjustment', 'cancellation',
                    'layaway', 'special order', 'gift card', 'store credit', 'loyalty', 'discount',
                    'promotion', 'markdown', 'coupon', 'employee', 'wholesale', 'bulk', 'web',
                    'pickup', 'delivery', 'subscription'
                ],
                'order_statuses': [
                    'pending', 'processing', 'picked', 'packed', 'shipped', 'in transit',
                    'delivered', 'completed', 'cancelled', 'returned', 'refunded', 'on hold',
                    'backorder', 'preorder', 'partial'
                ]
            },
            'healthcare': {
                'provider_types': [
                    'physician', 'doctor', 'md', 'do', 'nurse', 'rn', 'np', 'pa', 'therapist',
                    'specialist', 'surgeon', 'anesthesiologist', 'radiologist', 'pathologist',
                    'psychiatrist', 'psychologist', 'pharmacist', 'dentist', 'optometrist',
                    'chiropractor', 'midwife', 'dietitian'
                ],
                'facility_types': [
                    'hospital', 'clinic', 'practice', 'center', 'facility', 'office', 'lab',
                    'pharmacy', 'imaging', 'radiology', 'urgent care', 'emergency', 'ambulatory',
                    'outpatient', 'inpatient', 'acute', 'long-term', 'rehabilitation', 'skilled nursing',
                    'assisted living', 'home health', 'hospice'
                ],
                'insurance_types': [
                    'medicare', 'medicaid', 'commercial', 'private', 'hmo', 'ppo', 'epo', 'pos',
                    'high deductible', 'self-insured', 'self-funded', 'fully-insured', 'exchange',
                    'marketplace', 'cobra', 'group', 'individual', 'supplemental', 'vision',
                    'dental', 'workers comp', 'auto', 'liability', 'tricare', 'champva', 'va'
                ],
                'clinical_statuses': [
                    'active', 'inactive', 'resolved', 'chronic', 'acute', 'recurring', 'remission',
                    'controlled', 'uncontrolled', 'stable', 'unstable', 'improving', 'worsening',
                    'critical', 'serious', 'moderate', 'mild', 'pending', 'suspected', 'confirmed',
                    'ruled out', 'cancelled'
                ]
            },
            'finance': {
                'account_types': [
                    'checking', 'savings', 'money market', 'cd', 'certificate of deposit',
                    'retirement', 'ira', '401k', '403b', 'pension', 'investment', 'brokerage',
                    'credit', 'loan', 'mortgage', 'heloc', 'auto loan', 'student loan', 'personal loan',
                    'business', 'commercial', 'trust', 'custodial', 'escrow'
                ],
                'transaction_types': [
                    'deposit', 'withdrawal', 'transfer', 'payment', 'charge', 'credit', 'debit',
                    'purchase', 'refund', 'adjustment', 'fee', 'interest', 'dividend', 'distribution',
                    'contribution', 'direct deposit', 'ach', 'wire', 'check', 'cash', 'atm', 'pos',
                    'recurring', 'scheduled', 'automatic', 'manual', 'correction', 'reversal'
                ],
                'instrument_types': [
                    'stock', 'equity', 'etf', 'mutual fund', 'bond', 'treasury', 'municipal', 'corporate',
                    'option', 'future', 'forward', 'swap', 'forex', 'commodity', 'precious metal', 
                    'currency', 'cryptocurrency', 'reit', 'index', 'derivative', 'structured product',
                    'annuity', 'money market', 'cd', 'cdo', 'clo', 'abs', 'mbs'
                ],
                'account_statuses': [
                    'active', 'inactive', 'dormant', 'closed', 'frozen', 'restricted', 'blocked',
                    'on hold', 'overdrawn', 'negative', 'positive', 'zero balance', 'low balance',
                    'good standing', 'delinquent', 'default', 'charged off', 'collection',
                    'bankruptcy', 'settlement', 'paid off'
                ]
            },
            'manufacturing': {
                'material_types': [
                    'raw', 'processed', 'component', 'subassembly', 'assembly', 'finished good',
                    'wip', 'work in process', 'work in progress', 'semi-finished', 'consumable',
                    'packaging', 'spare part', 'maintenance', 'tool', 'equipment', 'fixture',
                    'jig', 'mold', 'die', 'chemical', 'adhesive', 'lubricant', 'hazardous'
                ],
                'process_types': [
                    'fabrication', 'machining', 'casting', 'molding', 'forming', 'welding',
                    'assembly', 'packaging', 'finishing', 'painting', 'coating', 'heat treatment',
                    'washing', 'testing', 'inspection', 'quality control', 'rework', 'repair',
                    'maintenance', 'setup', 'changeover', 'material handling', 'warehousing',
                    'shipping', 'receiving'
                ],
                'equipment_types': [
                    'machine', 'tool', 'press', 'lathe', 'mill', 'drill', 'grinder', 'saw',
                    'welder', 'robot', 'conveyor', 'crane', 'forklift', 'vehicle', 'computer',
                    'sensor', 'controller', 'plc', 'cnc', 'furnace', 'oven', 'mixer', 'pump',
                    'compressor', 'generator', 'transformer', 'motor', 'fan', 'filter'
                ],
                'quality_statuses': [
                    'passed', 'failed', 'pending', 'in process', 'on hold', 'quarantined', 'released',
                    'approved', 'rejected', 'scrapped', 'reworked', 'repaired', 'accepted', 'deviated',
                    'waived', 'conforming', 'nonconforming', 'major', 'minor', 'critical'
                ]
            }
        }
    
    def _initialize_domain_abbreviations(self):
        """Initialize common domain-specific abbreviations and acronyms"""
        return {
            'retail': {
                'pos': 'point of sale',
                'sku': 'stock keeping unit',
                'upc': 'universal product code',
                'ean': 'european article number',
                'gtin': 'global trade item number',
                'msrp': 'manufacturer suggested retail price',
                'map': 'minimum advertised price',
                'cogs': 'cost of goods sold',
                'gmroi': 'gross margin return on investment',
                'yoy': 'year over year',
                'mom': 'month over month',
                'wow': 'week over week',
                'dod': 'day over day',
                'aov': 'average order value',
                'ltv': 'lifetime value',
                'gm': 'gross margin',
                'rpm': 'retail price management',
                'bopis': 'buy online pickup in store',
                'bopus': 'buy online pick up at store',
                'boss': 'buy online ship to store',
                'ropis': 'reserve online pickup in store',
                'boris': 'buy online return in store',
                'oos': 'out of stock',
                'inv': 'inventory',
                'qty': 'quantity',
                'dept': 'department',
                'dc': 'distribution center',
                'rma': 'return merchandise authorization'
            },
            'healthcare': {
                'ehr': 'electronic health record',
                'emr': 'electronic medical record',
                'hie': 'health information exchange',
                'phi': 'protected health information',
                'hipaa': 'health insurance portability and accountability act',
                'cpt': 'current procedural terminology',
                'hcpcs': 'healthcare common procedure coding system',
                'icd': 'international classification of diseases',
                'drg': 'diagnosis related group',
                'snomed': 'systematized nomenclature of medicine',
                'loinc': 'logical observation identifiers names and codes',
                'ndc': 'national drug code',
                'npi': 'national provider identifier',
                'upin': 'unique physician identification number',
                'dea': 'drug enforcement administration',
                'mrn': 'medical record number',
                'pt': 'patient',
                'dx': 'diagnosis',
                'tx': 'treatment',
                'rx': 'prescription',
                'hmo': 'health maintenance organization',
                'ppo': 'preferred provider organization',
                'epo': 'exclusive provider organization',
                'pos': 'point of service',
                'snf': 'skilled nursing facility',
                'ltc': 'long term care',
                'los': 'length of stay',
                'ar': 'accounts receivable',
                'ub': 'uniform bill',
                'eob': 'explanation of benefits',
                'era': 'electronic remittance advice'
            },
            'finance': {
                'ach': 'automated clearing house',
                'atm': 'automated teller machine',
                'apr': 'annual percentage rate',
                'apy': 'annual percentage yield',
                'aum': 'assets under management',
                'cagr': 'compound annual growth rate',
                'cd': 'certificate of deposit',
                'cdo': 'collateralized debt obligation',
                'clo': 'collateralized loan obligation',
                'cma': 'cash management account',
                'cpi': 'consumer price index',
                'dda': 'demand deposit account',
                'eps': 'earnings per share',
                'etf': 'exchange traded fund',
                'fdic': 'federal deposit insurance corporation',
                'forex': 'foreign exchange',
                'ira': 'individual retirement account',
                'kpi': 'key performance indicator',
                'libor': 'london interbank offered rate',
                'loc': 'line of credit',
                'ltv': 'loan to value',
                'mbs': 'mortgage backed security',
                'nav': 'net asset value',
                'noa': 'notice of assignment',
                'nsf': 'non-sufficient funds',
                'otc': 'over the counter',
                'p2p': 'peer to peer',
                'piti': 'principal interest taxes insurance',
                'poc': 'proof of concept',
                'pos': 'point of sale',
                'reit': 'real estate investment trust',
                'roi': 'return on investment',
                'roth': 'roth individual retirement account',
                'roa': 'return on assets',
                'roe': 'return on equity',
                'sec': 'securities and exchange commission',
                'sip': 'systematic investment plan',
                'sma': 'separately managed account',
                'stp': 'straight through processing',
                'swift': 'society for worldwide interbank financial telecommunication',
                'ytd': 'year to date',
                'ytm': 'yield to maturity'
            },
            'manufacturing': {
                'bom': 'bill of materials',
                'cam': 'computer aided manufacturing',
                'cad': 'computer aided design',
                'cae': 'computer aided engineering',
                'cmm': 'coordinate measuring machine',
                'cnc': 'computer numerical control',
                'cots': 'commercial off the shelf',
                'capa': 'corrective action preventive action',
                'dfm': 'design for manufacturing',
                'dfa': 'design for assembly',
                'dft': 'design for testability',
                'eoq': 'economic order quantity',
                'erp': 'enterprise resource planning',
                'fmea': 'failure mode and effects analysis',
                'fifo': 'first in first out',
                'lifo': 'last in first out',
                'fpy': 'first pass yield',
                'haccp': 'hazard analysis critical control point',
                'iot': 'internet of things',
                'iqc': 'incoming quality control',
                'jit': 'just in time',
                'kanban': 'signboard or billboard',
                'kaizen': 'continuous improvement',
                'kpi': 'key performance indicator',
                'leed': 'leadership in energy and environmental design',
                'mes': 'manufacturing execution system',
                'mps': 'master production schedule',
                'mrp': 'material requirements planning',
                'msds': 'material safety data sheet',
                'mtbf': 'mean time between failures',
                'mttf': 'mean time to failure',
                'mttr': 'mean time to repair',
                'npi': 'new product introduction',
                'oee': 'overall equipment effectiveness',
                'pcb': 'printed circuit board',
                'pdm': 'product data management',
                'plm': 'product lifecycle management',
                'poka-yoke': 'mistake proofing',
                'ppap': 'production part approval process',
                'ppm': 'parts per million',
                'qa': 'quality assurance',
                'qc': 'quality control',
                'qms': 'quality management system',
                'rfid': 'radio frequency identification',
                'scada': 'supervisory control and data acquisition',
                'smed': 'single minute exchange of die',
                'spc': 'statistical process control',
                'tpm': 'total productive maintenance',
                'tqm': 'total quality management',
                'voc': 'voice of customer',
                'wip': 'work in process',
                'wms': 'warehouse management system'
            }
        }
    
    def get_domain_patterns(self, domain):
        """
        Get the domain-specific patterns for a given domain
        
        Args:
            domain (str): The domain to get patterns for (e.g., 'retail', 'healthcare')
            
        Returns:
            dict: Dictionary of patterns for the specified domain
        """
        return self.domain_patterns.get(domain, {})
    
    def get_domain_vocabulary(self, domain):
        """
        Get the domain-specific vocabulary for a given domain
        
        Args:
            domain (str): The domain to get vocabulary for
            
        Returns:
            dict: Dictionary of vocabulary terms for the specified domain
        """
        return self.domain_vocabularies.get(domain, {})
    
    def get_domain_abbreviations(self, domain):
        """
        Get the domain-specific abbreviations for a given domain
        
        Args:
            domain (str): The domain to get abbreviations for
            
        Returns:
            dict: Dictionary of abbreviations for the specified domain
        """
        return self.domain_abbreviations.get(domain, {})
    
    def get_all_domains(self):
        """
        Get a list of all available domains
        
        Returns:
            list: List of all available domains
        """
        return list(self.domain_patterns.keys())
    
    def is_domain_specific_column(self, column_name, domain):
        """
        Check if a column name matches any pattern in the specified domain
        
        Args:
            column_name (str): The column name to check
            domain (str): The domain to check against
            
        Returns:
            tuple: (bool, str) - Whether it's a match and the category if it is
        """
        domain_patterns = self.get_domain_patterns(domain)
        
        column_name_lower = column_name.lower()
        
        for category, patterns in domain_patterns.items():
            for pattern in patterns:
                if re.search(pattern, column_name_lower, re.IGNORECASE):
                    return True, category
        
        # Check abbreviations
        abbreviations = self.get_domain_abbreviations(domain)
        for abbr in abbreviations:
            if abbr in column_name_lower.split('_') or abbr in column_name_lower.split('-') or abbr == column_name_lower:
                return True, 'abbreviation'
        
        return False, None

    def detect_dominant_domain(self, column_names):
        """
        Detect the most likely domain based on a set of column names
        
        Args:
            column_names (list): List of column names to analyze
            
        Returns:
            tuple: (str, float) - The dominant domain and its confidence score
        """
        domain_scores = {domain: 0 for domain in self.get_all_domains()}
        
        for column in column_names:
            for domain in domain_scores.keys():
                is_match, category = self.is_domain_specific_column(column, domain)
                if is_match:
                    domain_scores[domain] += 1
        
        # Normalize scores
        total_matches = sum(domain_scores.values())
        if total_matches > 0:
            for domain in domain_scores:
                domain_scores[domain] = domain_scores[domain] / total_matches
        
        # Find domain with highest score
        dominant_domain = max(domain_scores.items(), key=lambda x: x[1])
        
        return dominant_domain[0], dominant_domain[1]
    
    def suggest_columns_by_domain(self, domain):
        """
        Suggest common columns for a specific domain
        
        Args:
            domain (str): The domain to suggest columns for
            
        Returns:
            dict: Dictionary of suggested columns by category
        """
        domain_patterns = self.get_domain_patterns(domain)
        
        suggestions = {}
        for category, patterns in domain_patterns.items():
            # Extract column name suggestions from the patterns
            column_suggestions = []
            for pattern in patterns:
                # Remove regex syntax and extract key terms
                simplified = re.sub(r'[\(\)\[\]\?\+\*\{\}\\]', '', pattern)
                simplified = re.sub(r'\|', ' ', simplified)
                simplified = re.sub(r'\\b', '', simplified)
                
                # Extract key terms
                for term in simplified.split():
                    if term not in ['and', 'or', 'in', 'of', 'the', 'a', 'an']:
                        column_suggestions.append(term.replace('_', '').replace('-', ''))
            
            # Filter to unique suggestions
            unique_suggestions = list(set(column_suggestions))
            
            if unique_suggestions:
                suggestions[category] = unique_suggestions[:5]  # Limit to 5 suggestions per category
        
        return suggestions

    def expand_abbreviation(self, abbreviation, domain=None):
        """
        Expand a domain-specific abbreviation
        
        Args:
            abbreviation (str): The abbreviation to expand
            domain (str, optional): The specific domain to check. If None, check all domains.
            
        Returns:
            str: The expanded abbreviation or the original if not found
        """
        abbreviation = abbreviation.lower()
        
        if domain:
            domain_abbrs = self.get_domain_abbreviations(domain)
            return domain_abbrs.get(abbreviation, abbreviation)
        else:
            # Check all domains
            for domain in self.get_all_domains():
                domain_abbrs = self.get_domain_abbreviations(domain)
                if abbreviation in domain_abbrs:
                    return domain_abbrs[abbreviation]
            
            return abbreviation

# Example usage
if __name__ == "__main__":
    # Initialize the domain knowledge base
    kb = DomainKnowledgeBase()
    
    # Test with some column names
    test_columns = [
        "order_id", 
        "product_sku", 
        "qty_ordered", 
        "unit_price", 
        "total_amount",
        "gmroi",
        "store_code"
    ]
    
    # Detect domain
    domain, confidence = kb.detect_dominant_domain(test_columns)
    print(f"Detected domain: {domain} (confidence: {confidence:.2f})")
    
    # Check domain-specific columns
    for column in test_columns:
        is_match, category = kb.is_domain_specific_column(column, domain)
        if is_match:
            print(f"Column '{column}' matches domain '{domain}' category '{category}'")
    
    # Expand abbreviations
    abbr_test = "gmroi"
    expanded = kb.expand_abbreviation(abbr_test, domain)
    print(f"Abbreviation '{abbr_test}' expands to: {expanded}")
    
    # Suggest columns for healthcare domain
    healthcare_suggestions = kb.suggest_columns_by_domain('healthcare')
    print("\nSuggested healthcare columns:")
    for category, columns in healthcare_suggestions.items():
        print(f"  {category}: {', '.join(columns)}")
