# Define comprehensive patterns for each column type
patterns = {
     'date': [
         r'^date$', r'^dt$', r'^day$', r'^time$', r'^timestamp$',
         r'^order.*date', r'^ord.*date', r'^ord.*dt',
         r'^invoice.*date', r'^invoice.*dt', r'^inv.*date',
         r'^transaction.*date', r'^trans.*date', r'^trans.*dt',
         r'^purchase.*date', r'^sale.*date', r'^delivery.*date', r'^ship.*date',
         r'^created.*date', r'^created.*at', r'^modified.*date',
         r'^year$', r'^month$', r'^quarter$', r'^period$'
     ],
     'sales': [
         r'^sales$', r'^sale$', r'^revenue$', r'^rev$', r'^amount$',
         r'^total.*sales', r'^sales.*amount', r'^sales.*amt', r'^sales.*val',
         r'^gross.*sales', r'^net.*sales', r'^sales.*value', r'^order.*value',
         r'^order.*amount', r'^order.*amt', r'^order.*val',
         r'^gmv$', r'^arr$', r'^transaction.*value', r'^trans.*value',
         r'^income$', r'^turnover$', r'^gp$', r'^gross.*profit',
         r'^net.*profit', r'^netprofit', r'^total$', r'^tot$',
         r'^revenue.*generated', r'^sales.*revenue', r'^total.*revenue'
     ],
     'quantity': [
         r'^quantity$', r'^qty$', r'^count$', r'^units?$', r'^volume$', r'^vol$',
         r'^order.*qty', r'^ord.*qty', r'^order.*count', r'^order.*units',
         r'^num.*items', r'^numitems', r'^item.*count', r'^itemcount',
         r'^pieces$', r'^pcs$', r'^units?.*sold', r'^quantity.*sold',
         r'^sales.*units', r'^sales.*qty', r'^sales.*count',
         r'^number.*of.*items', r'^num.*of.*units', r'^num.*of.*products'
     ],
     'price': [
         r'^price$', r'^unit.*price', r'^unitprice', r'^ppu$', r'^price.*per.*unit',
         r'^rate$', r'^cost$', r'^unit.*cost', r'^unitcost', r'^cost.*per.*unit',
         r'^item.*price', r'^item.*cost', r'^price.*per.*item',
         r'^avg.*price', r'^average.*price', r'^selling.*price', r'^sell.*price',
         r'^retail.*price', r'^list.*price', r'^base.*price', r'^unit.*rate',
         r'^rate.*per.*unit', r'^price.*point', r'^cost.*price'
     ],
     'discount_value': [
         r'^discount.*value', r'^discount.*amount', r'^discount.*amt',
         r'^disc.*value', r'^disc.*val', r'^disc.*amt', r'^disc.*amount',
         r'^discount.*dollars', r'^disc.*dollars', r'^discount.*total',
         r'^rebate.*amount', r'^rebate.*value', r'^rebate.*total',
         r'^promo.*value', r'^promo.*amount', r'^promotion.*value',
         r'^savings.*amount', r'^savings.*value', r'^deduction.*amount',
         r'^markdown.*amount', r'^markdown.*value', r'^reduction.*amount'
     ],
     'discount_pct': [
         r'^discount$', r'^disc$', r'^discount.*pct', r'^discount.*percent',
         r'^discount.*percentage', r'^disc.*pct', r'^disc.*percent',
         r'^discount.*rate', r'^disc.*rate', r'^rebate.*rate',
         r'^rebate.*pct', r'^rebate.*percentage', r'^promo.*rate',
         r'^promo.*discount', r'^promo.*pct', r'^promotion.*rate',
         r'^markdown.*percentage', r'^markdown.*rate', r'^markdown.*pct',
         r'^savings.*pct', r'^savings.*rate', r'^savings.*percentage',
         r'^reduction.*rate', r'^reduction.*pct', r'^disc.*off'
     ],
     'product': [
         r'^product$', r'^product.*id', r'^product.*name', r'^prod.*id',
         r'^prod.*name', r'^item$', r'^item.*id', r'^item.*name',
         r'^sku$', r'^upc$', r'^model$', r'^model.*number', r'^style$',
         r'^product.*code', r'^product.*number', r'^item.*number',
         r'^product.*desc', r'^item.*desc', r'^description',
         r'^category$', r'^subcategory$', r'^department$', r'^dept$',
         r'^brand$', r'^manufacturer$', r'^product.*type', r'^product.*family'
     ],
     'customer': [
         r'^customer$', r'^customer.*id', r'^cust.*id', r'^custid$', r'^custno$',
         r'^customer.*name', r'^cust.*name', r'^client$', r'^client.*id',
         r'^account$', r'^account.*id', r'^acct$', r'^acct.*id', r'^buyer$',
         r'^purchaser$', r'^customer.*number', r'^customer.*code',
         r'^account.*number', r'^cust.*type', r'^customer.*type',
         r'^customer.*segment', r'^segment$', r'^consumer$', r'^user$', r'^user.*id'
     ],
     'location': [
         r'^location$', r'^loc$', r'^store$', r'^store.*id', r'^storeid$',
         r'^store.*number', r'^branch$', r'^branch.*id', r'^site$', r'^site.*id',
         r'^region$', r'^area$', r'^territory$', r'^zone$', r'^district$',
         r'^country$', r'^state$', r'^city$', r'^province$', r'^county$',
         r'^zip$', r'^zipcode$', r'^postal.*code', r'^postalcode$',
         r'^market$', r'^market.*area', r'^geography$', r'^geo$', r'^ship.*to'
     ],
     'order_id': [
         r'^order.*id', r'^orderid$', r'^order.*no', r'^orderno$', r'^order.*number',
         r'^transaction.*id', r'^transactionid$', r'^trans.*id', r'^transid$',
         r'^invoice.*id', r'^invoiceid$', r'^invoice.*no', r'^invoiceno$', 
         r'^receipt.*id', r'^receipt.*no', r'^sale.*id', r'^sale.*no',
         r'^order.*code', r'^order.*ref', r'^reference.*id', r'^refid$',
         r'^confirmation.*id', r'^confirmation.*no', r'^purchase.*id'
     ]
 }
 
# Define lists of exact keywords for each column type for fuzzy matching
exact_keywords = {
     'date': [
         'date', 'dt', 'day', 'time', 'timestamp', 'orderdate', 'order_date',
         'invoicedate', 'invoice_date', 'transactiondate', 'transaction_date',
         'purchasedate', 'purchase_date', 'saledate', 'sale_date', 'shipdate', 
         'ship_date', 'deliverydate', 'delivery_date', 'createddate', 'created_date',
         'year', 'month', 'quarter', 'period'
     ],
     'sales': [
         'sales', 'sale', 'revenue', 'rev', 'amount', 'totalsales', 'total_sales',
         'salesamount', 'sales_amount', 'salesvalue', 'sales_value', 'grosssales',
         'gross_sales', 'netsales', 'net_sales', 'ordervalue', 'order_value',
         'orderamount', 'order_amount', 'gmv', 'arr', 'income', 'turnover', 'gp',
         'grossprofit', 'gross_profit', 'netprofit', 'net_profit', 'total', 'value'
     ],
     'quantity': [
         'quantity', 'qty', 'count', 'units', 'unit', 'volume', 'vol', 'orderqty',
         'order_qty', 'numitems', 'num_items', 'itemcount', 'item_count', 'pieces',
         'pcs', 'unitssold', 'units_sold', 'qtysold', 'qty_sold', 'unitcount', 'unit_count'
     ],
     'price': [
         'price', 'unitprice', 'unit_price', 'ppu', 'rate', 'cost', 'unitcost', 
         'unit_cost', 'itemprice', 'item_price', 'itemcost', 'item_cost', 'avgprice',
         'avg_price', 'sellingprice', 'selling_price', 'retailprice', 'retail_price',
         'listprice', 'list_price', 'baseprice', 'base_price', 'pricepoint', 'price_point'
     ],
     'discount_value': [
         'discountvalue', 'discount_value', 'discountamount', 'discount_amount',
         'discountamt', 'discount_amt', 'discvalue', 'disc_value', 'rebateamount',
         'rebate_amount', 'promovalue', 'promo_value', 'promotionamount', 'promotion_amount',
         'savingsamount', 'savings_amount', 'markdownamount', 'markdown_amount',
         'reductionamount', 'reduction_amount', 'discountdollars', 'discount_dollars'
     ],
     'discount_pct': [
         'discount', 'disc', 'discountpct', 'discount_pct', 'discountpercent',
         'discount_percent', 'discountpercentage', 'discount_percentage', 'discrate',
         'disc_rate', 'rebaterate', 'rebate_rate', 'promorate', 'promo_rate',
         'discountrate', 'discount_rate', 'markdownpct', 'markdown_pct', 'savingspct',
         'savings_pct', 'reductionrate', 'reduction_rate'
     ]
 }