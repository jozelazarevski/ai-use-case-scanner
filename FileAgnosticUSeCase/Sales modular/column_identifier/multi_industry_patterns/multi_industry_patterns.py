# Comprehensive pattern collection for column identification across industries
# This covers Sales, Finance, Marketing, Procurement, HR, Manufacturing, Healthcare, and more

class ComprehensivePatterns:
    """
    A comprehensive collection of patterns for identifying columns across multiple industries.
    This class provides pattern dictionaries that can be used by column identification systems
    to detect common column types in various business domains.
    """
    
    def __init__(self):
        """Initialize pattern collections for multiple industries"""
        # Initialize basic and advanced patterns
        self._init_basic_patterns()
        self._init_advanced_patterns()
        
        # Initialize industry-specific patterns
        self._init_sales_patterns()
        self._init_finance_patterns()
        self._init_marketing_patterns()
        self._init_procurement_patterns()
        self._init_hr_patterns()
        self._init_manufacturing_patterns()
        self._init_healthcare_patterns()
        self._init_education_patterns()
        self._init_ecommerce_patterns()
        self._init_logistics_patterns()
        self._init_real_estate_patterns()
    
    def _init_basic_patterns(self):
        """Initialize basic column patterns common across industries"""
        self.column_patterns = {
            # Time-related columns
            'date': ['date', 'time', 'day', 'month', 'year', 'dt', 'period', 'quarter', 'week', 'timestamp', 'datetime'],
            
            # Product-related columns
            'product': ['product', 'item', 'sku', 'model', 'goods', 'merchandise', 'prod', 'article', 'commodity', 'offering', 'stock', 'variant', 'asset'],
            
            # Quantity-related columns
            'quantity': ['quantity', 'qty', 'count', 'units', 'volume', 'amount', 'num', 'number', 'pcs', 'pieces', 'carton', 'bundle', 'lots', 'batch'],
            
            # Price-related columns
            'price': ['price', 'cost', 'rate', 'fee', 'charge', 'unit_price', 'unit price', '$', 'amount', 'value', 'msrp', 'rrp', 'list price', 'retail'],
            
            # Sales-related columns
            'sales': ['sales', 'revenue', 'total', 'amount', 'sum', 'income', 'proceeds', 'turnover', 'gross', 'billing', 'receipts', 'takings', 'sale amount'],
            
            # Customer-related columns
            'customer': ['customer', 'client', 'buyer', 'purchaser', 'consumer', 'cust', 'account', 'clientele', 'patron', 'prospect', 'shopper', 'guest', 'user'],
            
            # Location-related columns
            'location': ['location', 'store', 'region', 'area', 'territory', 'place', 'site', 'loc', 'country', 'city', 'state', 'province', 'county', 'zip', 'postal', 'address', 'geo'],
            
            # Category-related columns
            'category': ['category', 'type', 'group', 'class', 'department', 'segment', 'division', 'cat', 'kind', 'genre', 'family', 'collection', 'classification', 'taxonomy'],
            
            # Discount-related columns
            'discount': ['discount', 'reduction', 'deduction', 'markdown', 'rebate', 'promotion', 'promo', 'coupon', 'deal', 'offer', 'sale', 'voucher', 'allowance'],
            
            # Profit-related columns
            'profit': ['profit', 'margin', 'gain', 'earnings', 'net', 'return', 'yield', 'surplus', 'proceeds', 'roi', 'profitability', 'contribution', 'markup'],
            
            # Tax-related columns
            'tax': ['tax', 'vat', 'gst', 'duty', 'levy', 'impost', 'excise', 'tariff', 'tax rate', 'sales tax', 'withholding', 'customs'],
            
            # Payment-related columns
            'payment': ['payment', 'pay', 'transaction', 'settlement', 'remittance', 'transfer', 'deposit', 'installment', 'disbursement', 'expenditure', 'pay method'],
            
            # Shipping-related columns
            'shipping': ['shipping', 'delivery', 'transport', 'freight', 'shipment', 'dispatch', 'carrier', 'tracking', 'postage', 'carriage', 'forwarding', 'logistics'],
            
            # Order-related columns
            'order': ['order', 'purchase', 'requisition', 'booking', 'procurement', 'reservation', 'acquisition', 'po', 'cart', 'invoice', 'ticket', 'sales order'],
            
            # Status-related columns
            'status': ['status', 'state', 'condition', 'stage', 'phase', 'situation', 'disposition', 'standing', 'progress', 'step', 'flag', 'indicator'],
            
            # Time period-related columns
            'period': ['period', 'term', 'duration', 'span', 'interval', 'cycle', 'timeframe', 'epoch', 'era', 'fiscal', 'quarter', 'season'],
            
            # Employee-related columns
            'employee': ['employee', 'staff', 'personnel', 'worker', 'associate', 'member', 'colleague', 'rep', 'agent', 'consultant', 'contractor', 'resource'],
            
            # Department-related columns
            'department': ['department', 'dept', 'division', 'unit', 'section', 'branch', 'team', 'group', 'function', 'office', 'bureau', 'faculty'],
            
            # Budget-related columns
            'budget': ['budget', 'forecast', 'plan', 'allocation', 'appropriation', 'estimate', 'projection', 'target', 'limit', 'cap', 'fund', 'reserve'],
            
            # Cost-related columns
            'cost': ['cost', 'expense', 'expenditure', 'outlay', 'disbursement', 'charge', 'fee', 'price', 'payment', 'spend', 'investment', 'overhead'],
            
            # ID-related columns
            'id': ['id', 'identifier', 'key', 'code', 'reference', 'number', 'serial', 'uid', 'uuid', 'guid', 'index', 'primary key', 'pk'],
            
            # Name-related columns
            'name': ['name', 'title', 'label', 'description', 'desc', 'designation', 'caption', 'term', 'moniker', 'handle', 'alias', 'nomenclature'],
            
            # Email-related columns
            'email': ['email', 'e-mail', 'mail', 'electronic mail', 'email address', 'e-mail address', 'mailbox'],
            
            # Phone-related columns
            'phone': ['phone', 'telephone', 'tel', 'contact', 'mobile', 'cell', 'number', 'line', 'extension', 'fax', 'landline'],
            
            # Company-related columns
            'company': ['company', 'business', 'firm', 'enterprise', 'corporation', 'organization', 'establishment', 'institution', 'entity', 'agency', 'brand'],
            
            # Balance-related columns
            'balance': ['balance', 'total', 'amount', 'sum', 'remainder', 'rest', 'residue', 'closing', 'ending', 'net', 'outstanding'],
            
            # Activity-related columns
            'activity': ['activity', 'action', 'operation', 'task', 'process', 'procedure', 'function', 'transaction', 'event', 'exercise', 'movement'],
            
            # Currency-related columns
            'currency': ['currency', 'money', 'denomination', 'tender', 'cash', 'exchange', 'fx', 'coin', 'monetary'],
            
            # Rating-related columns
            'rating': ['rating', 'score', 'grade', 'rank', 'evaluation', 'assessment', 'appraisal', 'quality', 'stars', 'feedback', 'review'],
            
            # Weight-related columns
            'weight': ['weight', 'mass', 'load', 'heaviness', 'kg', 'pound', 'lb', 'oz', 'g', 'ton'],
            
            # Dimension-related columns
            'dimension': ['dimension', 'size', 'length', 'width', 'height', 'depth', 'diameter', 'radius', 'volume', 'area', 'cm', 'mm', 'inch', 'foot'],
            
            # Age-related columns
            'age': ['age', 'years', 'days', 'months', 'lifetime', 'seniority', 'generation', 'era', 'vintage', 'duration'],
            
            # Gender-related columns
            'gender': ['gender', 'sex', 'male', 'female', 'non-binary', 'm/f', 'gender identity'],
            
            # Percentage-related columns
            'percentage': ['percentage', 'percent', '%', 'proportion', 'ratio', 'rate', 'share', 'portion', 'fraction', 'quota', 'allocation'],
            
            # Frequency-related columns
            'frequency': ['frequency', 'rate', 'occurrence', 'repetition', 'recurrence', 'periodicity', 'regularity', 'rhythm', 'cadence', 'pace', 'interval'],
            
            # Description-related columns
            'description': ['description', 'desc', 'detail', 'specification', 'info', 'information', 'characteristic', 'attribute', 'feature', 'summary', 'overview'],
            
            # Account-related columns
            'account': ['account', 'acct', 'acc', 'ledger', 'book', 'registry', 'record', 'login', 'user', 'subscription', 'membership'],
            
            # Comment-related columns
            'comment': ['comment', 'note', 'remark', 'observation', 'annotation', 'statement', 'memo', 'feedback', 'opinion', 'review'],
            
            # URL-related columns
            'url': ['url', 'link', 'web', 'website', 'site', 'address', 'uri', 'hyperlink', 'domain', 'webpage'],
            
            # File-related columns
            'file': ['file', 'document', 'attachment', 'upload', 'download', 'image', 'picture', 'photo', 'video', 'audio', 'doc', 'pdf'],
            
            # Color-related columns
            'color': ['color', 'colour', 'hue', 'shade', 'tint', 'tone', 'pigment', 'dye', 'rgb', 'hex', 'palette'],
            
            # Material-related columns
            'material': ['material', 'fabric', 'substance', 'matter', 'element', 'ingredient', 'component', 'composition', 'makeup'],
            
            # Brand-related columns
            'brand': ['brand', 'make', 'manufacturer', 'producer', 'vendor', 'supplier', 'creator', 'label', 'marque', 'trademark'],
            
            # Language-related columns
            'language': ['language', 'lang', 'tongue', 'idiom', 'dialect', 'speech', 'linguistic', 'lingo', 'communication'],
            
            # Time-related columns
            'time': ['time', 'hour', 'minute', 'second', 'clock', 'duration', 'period', 'timespan', 'elapsed', 'hrs', 'mins', 'secs', 'millisecond'],
            
            # Priority-related columns
            'priority': ['priority', 'importance', 'urgency', 'criticality', 'rank', 'level', 'precedence', 'significance', 'weight', 'value'],
            
            # Project-related columns
            'project': ['project', 'initiative', 'program', 'venture', 'undertaking', 'task', 'assignment', 'endeavor', 'mission', 'job'],
            
            # Source-related columns
            'source': ['source', 'origin', 'provenance', 'derivation', 'root', 'cause', 'genesis', 'fountain', 'wellspring', 'fount'],
            
            # Target-related columns
            'target': ['target', 'goal', 'objective', 'aim', 'purpose', 'end', 'destination', 'mark', 'bullseye', 'quota'],
            
            # Risk-related columns
            'risk': ['risk', 'hazard', 'danger', 'threat', 'peril', 'vulnerability', 'exposure', 'liability', 'jeopardy', 'chance'],
            
            # Version-related columns
            'version': ['version', 'revision', 'edition', 'release', 'update', 'model', 'variant', 'iteration', 'build', 'vintage'],
            
            # Approval-related columns
            'approval': ['approval', 'authorization', 'sanction', 'permission', 'consent', 'endorsement', 'acceptance', 'confirmation', 'ratification'],
            
            # Failure-related columns
            'failure': ['failure', 'defect', 'fault', 'flaw', 'malfunction', 'breakdown', 'collapse', 'error', 'mistake', 'bug', 'glitch'],
            
            # Success-related columns
            'success': ['success', 'achievement', 'accomplishment', 'attainment', 'victory', 'triumph', 'win', 'realization', 'fulfillment'],
            
            # Duration-related columns
            'duration': ['duration', 'length', 'span', 'period', 'term', 'time', 'interval', 'stretch', 'extent', 'continuance'],
            
            # Inventory-related columns
            'inventory': ['inventory', 'stock', 'supply', 'store', 'stockpile', 'reserve', 'cache', 'repository', 'warehouse', 'surplus'],
            
            # Attendance-related columns
            'attendance': ['attendance', 'presence', 'turnout', 'appearance', 'participation', 'showing', 'company', 'audience', 'assembly'],
            
            # Weather-related columns
            'weather': ['weather', 'climate', 'condition', 'temperature', 'humidity', 'precipitation', 'forecast', 'atmospheric', 'meteorological']
        }
    
    def _init_advanced_patterns(self):
        """Initialize advanced patterns for more specific column identification"""
        self.advanced_patterns = {
            # Date-related advanced patterns
            'date': {
                'order_date': ['order date', 'purchase date', 'transaction date', 'sale date', 'order placement date', 'booking date'],
                'ship_date': ['ship date', 'shipping date', 'delivery date', 'sent date', 'dispatch date', 'fulfillment date'],
                'return_date': ['return date', 'refund date', 'exchange date', 'cancellation date'],
                'due_date': ['due date', 'deadline', 'payment due', 'expiration date', 'maturity date'],
                'created_date': ['created date', 'creation date', 'entered date', 'registration date', 'inception date'],
                'modified_date': ['modified date', 'updated date', 'change date', 'revision date', 'edit date'],
                'invoice_date': ['invoice date', 'billing date', 'statement date', 'charge date'],
                'start_date': ['start date', 'begin date', 'commence date', 'initiation date'],
                'end_date': ['end date', 'finish date', 'completion date', 'termination date', 'close date'],
                'birth_date': ['birth date', 'dob', 'date of birth', 'birthday', 'born date'],
                'hire_date': ['hire date', 'employment date', 'joining date', 'start date', 'onboarding date'],
                'received_date': ['received date', 'arrival date', 'intake date', 'accept date'],
                'issue_date': ['issue date', 'publication date', 'release date'],
                'effective_date': ['effective date', 'valid from date', 'activation date'],
                'expiry_date': ['expiry date', 'expiration date', 'end date', 'valid to date', 'termination date']
            },
            
            # Product-related advanced patterns
            'product': {
                'product_id': ['product id', 'item id', 'sku id', 'product code', 'item code', 'prod id', 'product #', 'product no', 'article number', 'item number'],
                'product_name': ['product name', 'item name', 'product description', 'item description', 'prod name', 'article name', 'merchandise name', 'goods name'],
                'product_type': ['product type', 'item type', 'product group', 'product category', 'prod type', 'item class', 'product classification'],
                'product_line': ['product line', 'line of products', 'product series', 'product family', 'collection', 'range', 'assortment'],
                'model_number': ['model number', 'model no', 'model #', 'model id', 'model code', 'style number', 'design number']
            },
            
            # Customer-related advanced patterns
            'customer': {
                'customer_id': ['customer id', 'client id', 'customer #', 'customer no', 'cust id', 'account id', 'customer code', 'client number', 'member id'],
                'customer_name': ['customer name', 'client name', 'buyer name', 'customer full name', 'account name', 'purchaser name', 'consumer name'],
                'customer_type': ['customer type', 'client type', 'account type', 'client category', 'customer segment', 'customer classification'],
                'customer_email': ['customer email', 'client email', 'account email', 'contact email', 'buyer email', 'email address'],
                'customer_phone': ['customer phone', 'client phone', 'contact phone', 'telephone', 'phone number', 'mobile', 'cell phone'],
                'customer_address': ['customer address', 'client address', 'shipping address', 'billing address', 'mailing address', 'delivery address']
            },
            
            # Price-related advanced patterns
            'price': {
                'unit_price': ['unit price', 'price per unit', 'item price', 'single price', 'per item price', 'each price'],
                'list_price': ['list price', 'catalog price', 'retail price', 'msrp', 'rrp', 'recommended price', 'suggested price'],
                'sale_price': ['sale price', 'discounted price', 'special price', 'promotional price', 'offer price', 'deal price'],
                'wholesale_price': ['wholesale price', 'trade price', 'dealer price', 'distributor price', 'partner price'],
                'cost_price': ['cost price', 'purchase price', 'acquisition price', 'buying price', 'supplier price', 'factory price']
            },
            
            # Quantity-related advanced patterns
            'quantity': {
                'ordered_quantity': ['ordered quantity', 'order qty', 'purchase quantity', 'requested quantity', 'demand quantity'],
                'shipped_quantity': ['shipped quantity', 'ship qty', 'delivered quantity', 'fulfilled quantity', 'dispatch quantity'],
                'received_quantity': ['received quantity', 'receipt qty', 'intake quantity', 'accepted quantity', 'arrived quantity'],
                'returned_quantity': ['returned quantity', 'return qty', 'refund quantity', 'exchange quantity', 'rejected quantity'],
                'available_quantity': ['available quantity', 'on hand quantity', 'in stock quantity', 'current quantity', 'inventory quantity'],
                'allocated_quantity': ['allocated quantity', 'reserved quantity', 'committed quantity', 'assigned quantity'],
                'minimum_quantity': ['minimum quantity', 'min qty', 'minimum order quantity', 'moq', 'threshold quantity', 'reorder point']
            },
            
            # Sales-related advanced patterns
            'sales': {
                'gross_sales': ['gross sales', 'total sales', 'sales before returns', 'pre-return sales', 'bruto sales'],
                'net_sales': ['net sales', 'sales after returns', 'sales after discounts', 'actual sales', 'real sales'],
                'sales_amount': ['sales amount', 'sales value', 'sales total', 'revenue amount', 'turnover amount'],
                'sales_volume': ['sales volume', 'sales quantity', 'units sold', 'volume of sales', 'sales count'],
                'sales_growth': ['sales growth', 'revenue growth', 'sales increase', 'growth rate', 'sales trend']
            },
            
            # Location-related advanced patterns
            'location': {
                'address': ['address', 'street address', 'mailing address', 'physical address', 'postal address', 'location address'],
                'city': ['city', 'town', 'municipality', 'locality', 'urban area', 'metropolitan area'],
                'state': ['state', 'province', 'region', 'territory', 'district', 'county', 'department'],
                'postal_code': ['postal code', 'zip code', 'zip', 'post code', 'pin code', 'postal area', 'postal district'],
                'country': ['country', 'nation', 'land', 'territory', 'state', 'republic', 'kingdom'],
                'longitude': ['longitude', 'long', 'x coordinate', 'meridian', 'easting'],
                'latitude': ['latitude', 'lat', 'y coordinate', 'parallel', 'northing']
            },
            
            # Order-related advanced patterns
            'order': {
                'order_id': ['order id', 'order number', 'order #', 'order no', 'purchase order', 'po number', 'transaction id', 'sale id'],
                'order_status': ['order status', 'order state', 'fulfillment status', 'processing status', 'delivery status'],
                'order_type': ['order type', 'order category', 'order class', 'purchase type', 'transaction type', 'order classification'],
                'order_amount': ['order amount', 'order total', 'order value', 'purchase amount', 'transaction amount', 'order sum'],
                'order_date': ['order date', 'purchase date', 'transaction date', 'date ordered', 'order placement date']
            },
            
            # Payment-related advanced patterns
            'payment': {
                'payment_id': ['payment id', 'payment number', 'payment reference', 'transaction id', 'payment code', 'receipt number'],
                'payment_method': ['payment method', 'payment type', 'payment mode', 'payment channel', 'payment form', 'pay method'],
                'payment_status': ['payment status', 'payment state', 'transaction status', 'payment condition', 'payment situation'],
                'payment_amount': ['payment amount', 'paid amount', 'transaction amount', 'payment sum', 'payment value', 'amount paid'],
                'payment_date': ['payment date', 'date paid', 'transaction date', 'payment received date', 'settlement date']
            },
            
            # Employee-related advanced patterns
            'employee': {
                'employee_id': ['employee id', 'staff id', 'personnel id', 'worker id', 'emp id', 'employee number', 'employee code'],
                'employee_name': ['employee name', 'staff name', 'personnel name', 'worker name', 'emp name', 'team member name'],
                'job_title': ['job title', 'position', 'title', 'role', 'designation', 'function', 'post', 'occupation'],
                'department': ['department', 'division', 'section', 'unit', 'team', 'group', 'branch', 'office'],
                'salary': ['salary', 'pay', 'wage', 'compensation', 'remuneration', 'earnings', 'income', 'stipend']
            },
            
            # Account-related advanced patterns
            'account': {
                'account_id': ['account id', 'account number', 'account #', 'account no', 'account code', 'acct id', 'acct no'],
                'account_name': ['account name', 'account title', 'account description', 'account label', 'account holder'],
                'account_type': ['account type', 'account category', 'account class', 'account group', 'account classification'],
                'account_status': ['account status', 'account state', 'account standing', 'account condition', 'account situation'],
                'account_balance': ['account balance', 'balance', 'current balance', 'available balance', 'ledger balance']
            },
            
            # Inventory-related advanced patterns
            'inventory': {
                'stock_level': ['stock level', 'inventory level', 'on hand', 'on hand quantity', 'available stock', 'current stock'],
                'reorder_point': ['reorder point', 'reorder level', 'minimum stock', 'threshold', 'par level', 'safety stock'],
                'safety_stock': ['safety stock', 'buffer stock', 'reserve stock', 'minimum inventory', 'stock cushion'],
                'stock_value': ['stock value', 'inventory value', 'stock worth', 'inventory cost', 'stock cost'],
                'stock_status': ['stock status', 'inventory status', 'stock condition', 'availability status', 'in stock status']
            },
            
            # Tax-related advanced patterns
            'tax': {
                'tax_rate': ['tax rate', 'tax percentage', 'tax percent', 'vat rate', 'gst rate', 'levy rate'],
                'tax_amount': ['tax amount', 'tax value', 'tax sum', 'vat amount', 'gst amount', 'tax total'],
                'tax_code': ['tax code', 'tax id', 'tax identifier', 'tax reference', 'tax category code'],
                'tax_type': ['tax type', 'tax category', 'tax class', 'tax classification', 'tax group', 'type of tax'],
                'tax_jurisdiction': ['tax jurisdiction', 'tax authority', 'tax region', 'tax territory', 'taxing body']
            },
            
            # Invoice-related advanced patterns
            'invoice': {
                'invoice_id': ['invoice id', 'invoice number', 'invoice #', 'invoice no', 'bill number', 'billing id'],
                'invoice_date': ['invoice date', 'bill date', 'billing date', 'date of invoice', 'invoice creation date'],
                'invoice_amount': ['invoice amount', 'invoice total', 'bill amount', 'billing amount', 'invoice sum'],
                'invoice_status': ['invoice status', 'billing status', 'payment status', 'invoice state', 'bill status'],
                'due_date': ['due date', 'payment due date', 'deadline', 'payable by date', 'payment date']
            }
        }
    
    def _init_sales_patterns(self):
        """Initialize sales-specific patterns"""
        self.sales_patterns = {
            # Sales KPIs
            'average_order_value': ['average order value', 'aov', 'avg order value', 'order average', 'mean order value'],
            'conversion_rate': ['conversion rate', 'conv rate', 'conversion percentage', 'sales conversion', 'lead conversion'],
            'customer_lifetime_value': ['customer lifetime value', 'clv', 'ltv', 'lifetime value', 'customer value'],
            'churn_rate': ['churn rate', 'attrition rate', 'customer loss rate', 'turnover rate', 'defection rate'],
            'retention_rate': ['retention rate', 'customer retention', 'client retention', 'repeat rate', 'loyalty rate'],
            
            # Sales pipeline
            'lead_source': ['lead source', 'source of lead', 'lead origin', 'prospect source', 'referral source'],
            'opportunity': ['opportunity', 'sales opportunity', 'deal', 'potential sale', 'prospect', 'lead'],
            'pipeline_stage': ['pipeline stage', 'sales stage', 'deal stage', 'funnel stage', 'sales phase'],
            'win_probability': ['win probability', 'close probability', 'success probability', 'probability of sale', 'likelihood'],
            'forecast': ['forecast', 'sales forecast', 'revenue forecast', 'prediction', 'projection', 'estimation'],
            
            # Sales performance
            'quota': ['quota', 'sales quota', 'target', 'goal', 'objective', 'sales target', 'revenue target'],
            'attainment': ['attainment', 'achievement', 'performance', 'quota attainment', 'target attainment'],
            'commission': ['commission', 'sales commission', 'bonus', 'incentive', 'compensation', 'reward'],
            'territory': ['territory', 'sales territory', 'region', 'area', 'district', 'zone', 'sector'],
            'segment': ['segment', 'market segment', 'customer segment', 'client segment', 'buyer segment']
        }
    
    def _init_finance_patterns(self):
        """Initialize finance-specific patterns"""
        self.finance_patterns = {
            # Accounting terms
            'asset': ['asset', 'assets', 'fixed asset', 'current asset', 'tangible asset', 'intangible asset'],
            'liability': ['liability', 'liabilities', 'debt', 'obligation', 'current liability', 'long-term liability'],
            'equity': ['equity', 'owner equity', 'shareholder equity', 'net worth', 'capital', 'retained earnings'],
            'revenue': ['revenue', 'income', 'earnings', 'proceeds', 'turnover', 'top line', 'gross revenue'],
            'expense': ['expense', 'cost', 'expenditure', 'outlay', 'payment', 'charge', 'expense item'],
            
            # Financial ratios
            'profit_margin': ['profit margin', 'margin', 'net margin', 'gross margin', 'operating margin', 'markup'],
            'roi': ['roi', 'return on investment', 'rate of return', 'return', 'yield', 'investment return'],
            'ebitda': ['ebitda', 'earnings before interest taxes depreciation amortization', 'operating earnings', 'operating income'],
            'cash_flow': ['cash flow', 'cashflow', 'operating cash flow', 'free cash flow', 'cash inflow', 'cash outflow'],
            'working_capital': ['working capital', 'net working capital', 'operating capital', 'current capital', 'circulating capital'],
            
            # Banking terms
            'account_number': ['account number', 'account #', 'account no', 'bank account', 'acct number', 'acct no'],
            'routing_number': ['routing number', 'routing #', 'aba number', 'transit number', 'routing code'],
            'interest_rate': ['interest rate', 'rate of interest', 'interest percentage', 'annual rate', 'apr', 'apy'],
            'principal': ['principal', 'principal amount', 'original amount', 'base amount', 'capital sum', 'loan amount'],
            'maturity_date': ['maturity date', 'due date', 'expiry date', 'term end date', 'final payment date'],
            
            # Investment terms
            'portfolio': ['portfolio', 'investment portfolio', 'asset portfolio', 'security portfolio', 'holdings'],
            'dividend': ['dividend', 'div', 'payout', 'distribution', 'profit sharing', 'capital distribution'],
            'stock_price': ['stock price', 'share price', 'market price', 'equity price', 'trading price', 'quote'],
            'market_value': ['market value', 'market cap', 'capitalization', 'valuation', 'worth', 'price'],
            'volatility': ['volatility', 'price volatility', 'market volatility', 'fluctuation', 'variation', 'standard deviation']
        }
    
    def _init_marketing_patterns(self):
        """Initialize marketing-specific patterns"""
        self.marketing_patterns = {
            # Digital marketing
            'impression': ['impression', 'view', 'ad impression', 'display', 'exposure', 'visibility', 'ad view'],
            'click': ['click', 'ad click', 'link click', 'tap', 'selection', 'hit', 'click-through'],
            'ctr': ['ctr', 'click through rate', 'click rate', 'click percentage', 'clickthrough rate'],
            'conversion': ['conversion', 'goal completion', 'action', 'completed action', 'response', 'acquisition'],
            'bounce_rate': ['bounce rate', 'exit rate', 'abandonment rate', 'single page visit rate', 'drop-off rate'],
            
            # Campaign metrics
            'campaign_name': ['campaign name', 'campaign title', 'promotion name', 'marketing campaign', 'ad campaign'],
            'campaign_id': ['campaign id', 'campaign code', 'campaign number', 'campaign reference', 'promotion id'],
            'audience': ['audience', 'target audience', 'target market', 'target group', 'target segment', 'market segment'],
            'channel': ['channel', 'marketing channel', 'media channel', 'distribution channel', 'communication channel'],
            'campaign_cost': ['campaign cost', 'marketing cost', 'promotion cost', 'ad spend', 'media spend'],
            
            # Content marketing
            'content_type': ['content type', 'content format', 'media type', 'asset type', 'creative type'],
            'engagement': ['engagement', 'interaction', 'involvement', 'participation', 'activity', 'social engagement'],
            'reach': ['reach', 'audience reach', 'exposure', 'coverage', 'visibility', 'impressions'],
            'open_rate': ['open rate', 'email open rate', 'message open rate', 'view rate', 'open percentage'],
            'click_rate': ['click rate', 'email click rate', 'link click rate', 'message click rate', 'click percentage'],
            
            # Customer metrics
            'acquisition_cost': ['acquisition cost', 'cac', 'cost per acquisition', 'customer cost', 'cost per customer'],
            'nps': ['nps', 'net promoter score', 'promoter score', 'recommendation score', 'loyalty metric'],
            'sentiment': ['sentiment', 'customer sentiment', 'opinion', 'attitude', 'feeling', 'perception'],
            'satisfaction': ['satisfaction', 'customer satisfaction', 'csat', 'satisfaction score', 'happiness score'],
            'persona': ['persona', 'buyer persona', 'customer persona', 'user persona', 'target persona', 'avatar']
        }
    
    def _init_procurement_patterns(self):
        """Initialize procurement-specific patterns"""
        self.procurement_patterns = {
            # Purchase order terms
            'po_number': ['po number', 'purchase order number', 'po #', 'po id', 'order number', 'requisition number'],
            'requisition': ['requisition', 'req', 'purchase requisition', 'material requisition', 'pr number', 'purchase request'],
            'vendor': ['vendor', 'supplier', 'provider', 'seller', 'merchant', 'contractor', 'source'],
            'vendor_id': ['vendor id', 'supplier id', 'vendor code', 'supplier code', 'vendor number', 'supplier number'],
            'buyer': ['buyer', 'purchasing agent', 'procurement specialist', 'purchasing officer', 'sourcing manager'],
            
            # Purchasing metrics
            'lead_time': ['lead time', 'delivery time', 'procurement time', 'ordering time', 'supply time', 'turnaround time'],
            'delivery_date': ['delivery date', 'receipt date', 'expected date', 'arrival date', 'eta', 'expected delivery'],
            'order_quantity': ['order quantity', 'purchase quantity', 'buy quantity', 'lot size', 'order size', 'procurement quantity'],
            'unit_of_measure': ['unit of measure', 'uom', 'measurement unit', 'unit', 'measure', 'quantity unit'],
            'reorder_point': ['reorder point', 'reorder level', 'minimum quantity', 'trigger point', 'replenishment point'],
            
            # Contract terms
            'contract_id': ['contract id', 'contract number', 'agreement id', 'contract reference', 'legal agreement number'],
            'contract_term': ['contract term', 'agreement term', 'contract duration', 'contract period', 'agreement period'],
            'payment_term': ['payment term', 'payment terms', 'payment condition', 'payment period', 'credit term'],
            'warranty': ['warranty', 'guarantee', 'assurance', 'warranty period', 'guarantee term', 'coverage'],
            'term_condition': ['term and condition', 'terms and conditions', 't&c', 'condition', 'stipulation', 'provision'],
            
            # Supplier management
            'supplier_rating': ['supplier rating', 'vendor rating', 'supplier score', 'vendor performance', 'supplier evaluation'],
            'supply_chain': ['supply chain', 'value chain', 'supply network', 'logistics chain', 'distribution network'],
            'sourcing': ['sourcing', 'procurement sourcing', 'strategic sourcing', 'supplier sourcing', 'vendor sourcing'],
            'compliance': ['compliance', 'regulatory compliance', 'rule adherence', 'conformity', 'regulation compliance'],
            'sustainability': ['sustainability', 'sustainable sourcing', 'green procurement', 'eco-friendly', 'environmental impact']
        }
    
    def _init_hr_patterns(self):
        """Initialize human resources-specific patterns"""
        self.hr_patterns = {
            # Employee data
            'employee_number': ['employee number', 'employee #', 'emp no', 'staff number', 'personnel number', 'worker id'],
            'first_name': ['first name', 'given name', 'forename', 'christian name', 'personal name', 'first'],
            'last_name': ['last name', 'surname', 'family name', 'second name', 'paternal name', 'last'],
            'ssn': ['ssn', 'social security number', 'social security', 'social insurance number', 'national id', 'tax id'],
            'hire_date': ['hire date', 'employment date', 'start date', 'joining date', 'onboarding date', 'date of hire'],
            
            # Compensation
            'salary': ['salary', 'base salary', 'annual salary', 'basic salary', 'pay', 'wage', 'remuneration'],
            'hourly_rate': ['hourly rate', 'hourly wage', 'rate per hour', 'hourly pay', 'per hour rate', 'hour rate'],
            'bonus': ['bonus', 'incentive', 'performance bonus', 'reward', 'premium', 'gratuity', 'perk'],
            'benefits': ['benefits', 'employee benefits', 'perks', 'fringe benefits', 'compensation benefits', 'benefit package'],
            'deduction': ['deduction', 'withholding', 'pay deduction', 'salary deduction', 'contribution', 'tax deduction'],
            
            # Performance management
            'performance_rating': ['performance rating', 'performance score', 'evaluation score', 'appraisal rating', 'review score'],
            'goal': ['goal', 'objective', 'target', 'aim', 'performance goal', 'development goal', 'okr'],
            'competency': ['competency', 'skill', 'capability', 'proficiency', 'ability', 'talent', 'expertise'],
            'review_date': ['review date', 'evaluation date', 'appraisal date', 'assessment date', 'performance review date'],
            'feedback': ['feedback', 'review feedback', 'performance feedback', 'comment', 'input', 'critique', 'assessment'],
            
            # Time and attendance
            'time_in': ['time in', 'clock in', 'start time', 'arrival time', 'check in', 'punch in', 'log in'],
            'time_out': ['time out', 'clock out', 'end time', 'departure time', 'check out', 'punch out', 'log out'],
            'hours_worked': ['hours worked', 'work hours', 'labor hours', 'working hours', 'billable hours', 'time worked'],
            'overtime': ['overtime', 'ot', 'overtime hours', 'extra hours', 'additional hours', 'extended hours'],
            'absence': ['absence', 'time off', 'leave', 'day off', 'vacation', 'sick leave', 'personal time']
        }
    
    def _init_manufacturing_patterns(self):
        """Initialize manufacturing-specific patterns"""
        self.manufacturing_patterns = {
            # Production
            'production_order': ['production order', 'work order', 'manufacturing order', 'job order', 'process order', 'batch order'],
            'production_line': ['production line', 'assembly line', 'manufacturing line', 'processing line', 'factory line'],
            'yield': ['yield', 'production yield', 'output rate', 'efficiency rate', 'process yield', 'manufacturing yield'],
            'downtime': ['downtime', 'idle time', 'stoppage', 'production halt', 'machine downtime', 'equipment downtime'],
            'cycle_time': ['cycle time', 'process time', 'production time', 'manufacturing time', 'throughput time', 'takt time'],
            
            # Materials
            'raw_material': ['raw material', 'input material', 'feedstock', 'starting material', 'base material', 'ingredient'],
            'bill_of_materials': ['bill of materials', 'bom', 'materials list', 'parts list', 'component list', 'ingredient list'],
            'work_in_progress': ['work in progress', 'wip', 'work in process', 'in-process inventory', 'partially completed'],
            'finished_good': ['finished good', 'end product', 'final product', 'completed item', 'manufactured good'],
            'scrap': ['scrap', 'waste', 'reject', 'discarded material', 'unusable output', 'manufacturing waste'],
            
            # Equipment
            'machine': ['machine', 'equipment', 'apparatus', 'device', 'tool', 'machinery', 'manufacturing equipment'],
            'machine_id': ['machine id', 'equipment id', 'machine number', 'machine code', 'equipment number', 'tool id'],
            'capacity': ['capacity', 'production capacity', 'manufacturing capacity', 'output capacity', 'throughput capacity'],
            'utilization': ['utilization', 'machine utilization', 'equipment utilization', 'capacity utilization', 'usage rate'],
            'maintenance': ['maintenance', 'repair', 'servicing', 'upkeep', 'preventive maintenance', 'equipment maintenance'],
            
            # Quality
            'defect': ['defect', 'flaw', 'fault', 'imperfection', 'nonconformity', 'defective', 'reject'],
            'defect_rate': ['defect rate', 'error rate', 'fault rate', 'rejection rate', 'failure rate', 'quality incident rate'],
            'inspection': ['inspection', 'quality check', 'examination', 'verification', 'quality control', 'qc check'],
            'tolerance': ['tolerance', 'allowable variation', 'acceptable deviation', 'specification limit', 'quality tolerance'],
            'specification': ['specification', 'spec', 'technical specification', 'product spec', 'quality specification']
        }
    
    def _init_healthcare_patterns(self):
        """Initialize healthcare-specific patterns"""
        self.healthcare_patterns = {
            # Patient information
            'patient_id': ['patient id', 'patient number', 'medical record number', 'mrn', 'chart number', 'patient identifier'],
            'diagnosis': ['diagnosis', 'medical diagnosis', 'clinical diagnosis', 'diagnostic code', 'condition', 'ailment'],
            'treatment': ['treatment', 'therapy', 'intervention', 'procedure', 'regimen', 'care', 'medical treatment'],
            'medication': ['medication', 'medicine', 'drug', 'pharmaceutical', 'prescription', 'remedy', 'medicament'],
            'allergy': ['allergy', 'allergic reaction', 'sensitivity', 'intolerance', 'adverse reaction', 'hypersensitivity'],
            
            # Clinical data
            'vital_sign': ['vital sign', 'vitals', 'life sign', 'clinical sign', 'physiological stat', 'body measurement'],
            'blood_pressure': ['blood pressure', 'bp', 'arterial pressure', 'systolic pressure', 'diastolic pressure'],
            'heart_rate': ['heart rate', 'pulse', 'pulse rate', 'heart beat', 'cardiac rate', 'bpm', 'beats per minute'],
            'temperature': ['temperature', 'body temperature', 'temp', 'fever', 'hypothermia', 'celsius', 'fahrenheit'],
            'blood_glucose': ['blood glucose', 'blood sugar', 'glucose level', 'glycemia', 'sugar level', 'glucose reading'],
            
            # Administrative
            'admission_date': ['admission date', 'date of admission', 'hospital admission', 'entry date', 'check-in date'],
            'discharge_date': ['discharge date', 'date of discharge', 'hospital discharge', 'release date', 'check-out date'],
            'physician': ['physician', 'doctor', 'medical doctor', 'md', 'clinician', 'practitioner', 'attending'],
            'department': ['department', 'ward', 'unit', 'clinic', 'specialty', 'service', 'medical department'],
            'insurance': ['insurance', 'health insurance', 'medical insurance', 'coverage', 'insurance plan', 'policy']
        }
    
    def _init_education_patterns(self):
        """Initialize education-specific patterns"""
        self.education_patterns = {
            # Student information
            'student_id': ['student id', 'student number', 'learner id', 'pupil id', 'student identifier', 'enrollment number'],
            'grade': ['grade', 'mark', 'score', 'grading', 'scholastic mark', 'academic grade', 'test score'],
            'course': ['course', 'class', 'subject', 'module', 'unit', 'study', 'curriculum', 'educational program'],
            'enrollment': ['enrollment', 'registration', 'admission', 'entry', 'matriculation', 'subscription', 'signup'],
            'graduation': ['graduation', 'completion', 'commencement', 'finishing', 'conferral', 'award ceremony'],
            
            # Academic
            'degree': ['degree', 'qualification', 'academic degree', 'diploma', 'certificate', 'academic award'],
            'major': ['major', 'concentration', 'specialization', 'focus area', 'field of study', 'discipline'],
            'gpa': ['gpa', 'grade point average', 'academic average', 'grade average', 'quality point average'],
            'credit': ['credit', 'credit hour', 'course credit', 'unit', 'academic credit', 'study credit'],
            'semester': ['semester', 'term', 'academic term', 'session', 'quarter', 'period', 'academic period'],
            
            # Faculty and staff
            'instructor': ['instructor', 'teacher', 'professor', 'lecturer', 'faculty member', 'educator', 'academic'],
            'department': ['department', 'faculty', 'division', 'school', 'academic unit', 'institute', 'college'],
            'advisor': ['advisor', 'adviser', 'mentor', 'counselor', 'supervisor', 'tutor', 'academic advisor'],
            'program': ['program', 'curriculum', 'course of study', 'academic program', 'study program', 'degree program'],
            'accreditation': ['accreditation', 'certification', 'recognition', 'endorsement', 'validation', 'approval']
        }
    
    def _init_ecommerce_patterns(self):
        """Initialize e-commerce-specific patterns"""
        self.ecommerce_patterns = {
            # Product data
            'sku': ['sku', 'stock keeping unit', 'product code', 'item number', 'article number', 'product reference'],
            'upc': ['upc', 'universal product code', 'barcode', 'product barcode', 'gtin-12', 'gtin12'],
            'ean': ['ean', 'european article number', 'international article number', 'gtin-13', 'gtin13', 'barcode'],
            'asin': ['asin', 'amazon standard identification number', 'amazon id', 'amazon product id'],
            'mpn': ['mpn', 'manufacturer part number', 'part number', 'manufacturer number', 'oem part number'],
            
            # Shopping experience
            'cart': ['cart', 'shopping cart', 'basket', 'shopping basket', 'trolley', 'shopping bag', 'order'],
            'wishlist': ['wishlist', 'wish list', 'favorites', 'saved items', 'saved for later', 'saved products'],
            'checkout': ['checkout', 'check-out', 'payment process', 'payment page', 'order completion', 'purchase process'],
            'abandoned_cart': ['abandoned cart', 'cart abandonment', 'incomplete purchase', 'unfinished order', 'dropped cart'],
            'add_to_cart': ['add to cart', 'add to basket', 'add item', 'place in cart', 'put in basket', 'add to bag'],
            
            # Metrics
            'conversion_rate': ['conversion rate', 'e-commerce conversion', 'sales conversion', 'cart to purchase', 'browse to buy'],
            'aov': ['aov', 'average order value', 'basket value', 'cart value', 'transaction value', 'order average'],
            'ctr': ['ctr', 'click through rate', 'click rate', 'product click rate', 'listing click rate', 'ad click rate'],
            'bounce_rate': ['bounce rate', 'exit rate', 'abandonment rate', 'single page visit', 'quick exit rate'],
            'cart_abandonment': ['cart abandonment', 'abandonment rate', 'cart drop-off', 'shopping cart abandonment'],
            
            # Fulfillment
            'tracking_number': ['tracking number', 'tracking code', 'shipment number', 'tracking id', 'package tracking'],
            'shipping_method': ['shipping method', 'delivery method', 'shipment type', 'delivery option', 'shipping option'],
            'fulfillment': ['fulfillment', 'order fulfillment', 'shipping process', 'order processing', 'shipping fulfillment'],
            'delivery_status': ['delivery status', 'shipping status', 'order status', 'fulfillment status', 'tracking status'],
            'return': ['return', 'product return', 'item return', 'refund request', 'exchange request', 'return merchandise']
        }
    
    def _init_logistics_patterns(self):
        """Initialize logistics-specific patterns"""
        self.logistics_patterns = {
            # Shipping
            'tracking': ['tracking', 'tracking number', 'tracking id', 'shipment tracking', 'package tracking', 'trace code'],
            'carrier': ['carrier', 'shipping carrier', 'freight carrier', 'shipping company', 'transport provider', 'logistics provider'],
            'mode': ['mode', 'transport mode', 'shipping mode', 'delivery mode', 'freight mode', 'transportation type'],
            'origin': ['origin', 'source', 'starting point', 'place of origin', 'departure point', 'shipping origin'],
            'destination': ['destination', 'delivery location', 'consignee location', 'arrival point', 'delivery address', 'shipping destination'],
            
            # Freight
            'weight': ['weight', 'gross weight', 'net weight', 'cargo weight', 'shipment weight', 'freight weight'],
            'dimension': ['dimension', 'size', 'measurement', 'length', 'width', 'height', 'volume', 'cubic measurement'],
            'packaging': ['packaging', 'packing', 'package type', 'container type', 'wrapping', 'carton', 'box'],
            'freight_class': ['freight class', 'shipping class', 'transportation class', 'nmfc', 'freight classification'],
            'hazmat': ['hazmat', 'hazardous material', 'dangerous goods', 'hazardous goods', 'restricted material'],
            
            # Documents
            'bill_of_lading': ['bill of lading', 'bol', 'b/l', 'waybill', 'consignment note', 'transport document'],
            'commercial_invoice': ['commercial invoice', 'invoice', 'shipping invoice', 'export invoice', 'sales invoice'],
            'packing_list': ['packing list', 'pack list', 'packaging list', 'shipping list', 'contents list', 'item list'],
            'customs_declaration': ['customs declaration', 'customs form', 'declaration form', 'entry form', 'customs document'],
            'certificate_of_origin': ['certificate of origin', 'coo', 'origin certificate', 'proof of origin', 'country of origin certificate'],
            
            # Facility
            'warehouse': ['warehouse', 'distribution center', 'fulfillment center', 'storage facility', 'depot', 'storehouse'],
            'dock': ['dock', 'loading dock', 'shipping dock', 'receiving dock', 'bay', 'platform', 'berth'],
            'aisle': ['aisle', 'passageway', 'corridor', 'lane', 'walkway', 'warehouse aisle', 'storage aisle'],
            'rack': ['rack', 'shelving', 'storage rack', 'pallet rack', 'racking', 'shelves', 'storage system'],
            'bin': ['bin', 'container', 'storage bin', 'box', 'tote', 'receptacle', 'storage container']
        }
    
    def _init_real_estate_patterns(self):
        """Initialize real estate-specific patterns"""
        self.real_estate_patterns = {
            # Property information
            'property_id': ['property id', 'property number', 'listing id', 'real estate id', 'home id', 'estate id'],
            'address': ['address', 'property address', 'location', 'street address', 'postal address', 'mailing address'],
            'property_type': ['property type', 'home type', 'building type', 'house type', 'residential type', 'commercial type'],
            'square_footage': ['square footage', 'area', 'floor area', 'living area', 'building size', 'square meters'],
            'lot_size': ['lot size', 'land size', 'acreage', 'property size', 'parcel size', 'land area'],
            
            # Financial information
            'list_price': ['list price', 'asking price', 'offering price', 'property price', 'listing price', 'advertised price'],
            'sale_price': ['sale price', 'sold price', 'closing price', 'final price', 'purchase price', 'transaction price'],
            'assessment': ['assessment', 'property assessment', 'tax assessment', 'assessed value', 'valuation', 'appraisal'],
            'mortgage': ['mortgage', 'home loan', 'property loan', 'housing loan', 'real estate loan', 'loan amount'],
            'rent': ['rent', 'rental price', 'monthly rent', 'lease amount', 'rental fee', 'rental rate'],
            
            # Features
            'bedrooms': ['bedrooms', 'beds', 'bedroom count', 'number of bedrooms', 'bedroom quantity', 'br'],
            'bathrooms': ['bathrooms', 'baths', 'bathroom count', 'number of bathrooms', 'bathroom quantity', 'ba'],
            'parking': ['parking', 'garage', 'parking space', 'car park', 'carport', 'parking spot'],
            'year_built': ['year built', 'construction year', 'build year', 'age', 'construction date', 'year of construction'],
            'amenities': ['amenities', 'features', 'facilities', 'conveniences', 'perks', 'property features']
        }
    
    def get_all_patterns(self):
        """
        Get all patterns from all categories combined into a single dictionary.
        
        Returns:
            dict: Combined dictionary of all patterns
        """
        all_patterns = self.column_patterns.copy()
        
        # Add industry-specific patterns if they exist
        for industry_attr in [
            'sales_patterns', 'finance_patterns', 'marketing_patterns', 'procurement_patterns',
            'hr_patterns', 'manufacturing_patterns', 'healthcare_patterns', 'education_patterns',
            'ecommerce_patterns', 'logistics_patterns', 'real_estate_patterns'
        ]:
            if hasattr(self, industry_attr):
                industry_patterns = getattr(self, industry_attr)
                for key, patterns in industry_patterns.items():
                    if key not in all_patterns:
                        all_patterns[key] = patterns
                    else:
                        # Merge with existing patterns, avoiding duplicates
                        existing_patterns = set(all_patterns[key])
                        for pattern in patterns:
                            if pattern not in existing_patterns:
                                all_patterns[key].append(pattern)
        
        return all_patterns
    
    def get_industry_patterns(self, industry):
        """
        Get patterns specific to a particular industry.
        
        Args:
            industry (str): The industry name (sales, finance, marketing, etc.)
            
        Returns:
            dict: Dictionary of patterns for the specified industry
        """
        industry_attr = f"{industry.lower()}_patterns"
        if hasattr(self, industry_attr):
            return getattr(self, industry_attr)
        else:
            return {}
    
    def get_advanced_patterns(self, column_type=None):
        """
        Get advanced patterns for column detection, optionally filtered by column type.
        
        Args:
            column_type (str, optional): Specific column type to get patterns for
            
        Returns:
            dict: Dictionary of advanced patterns
        """
        if column_type:
            if column_type in self.advanced_patterns:
                return {column_type: self.advanced_patterns[column_type]}
            else:
                return {}
        else:
            return self.advanced_patterns