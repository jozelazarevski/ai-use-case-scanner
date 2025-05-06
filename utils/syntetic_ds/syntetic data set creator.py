import pandas as pd
import numpy as np
import random
import os

# Define the number of datasets to generate
num_datasets = 50

# Define the base directory where datasets will be saved
base_directory = "synthetic_data"
os.makedirs(base_directory, exist_ok=True)  # Create the directory if it doesn't exist

# Define a list of well-known ERP/CRM systems
systems = [
    "SAP_ERP", "Salesforce_CRM", "Oracle_EBS", "Microsoft_Dynamics_365", "NetSuite_ERP",
    "HubSpot_CRM", "Zoho_CRM", "Infor_ERP", "Sage_Intacct", "Workday_ERP",
    "SAP_S4HANA", "Salesforce_Sales_Cloud", "Oracle_Fusion", "Microsoft_Dynamics_Sales",
    "Odoo_ERP_CRM", "SugarCRM", "SuiteCRM", "Epicor_ERP", "Acumatica_ERP", "Priority_ERP",
    "Teamcenter_PLM", "Kinaxis_RapidResponse", "Coupa_Procure_to_Pay", "ServiceNow_CSM", "Zendesk_Sell",
    "Freshsales_CRM", "Pipedrive_CRM", "Monday_CRM", "Vtiger_CRM", "Insightly_CRM",
    "Maximizer_CRM", "Really_Simple_Systems_CRM", "Gold-Vision_CRM", "Raynet_CRM", "Daylite_CRM",
    "InfoFlo_CRM", "Base_CRM", "Nutshell_CRM", "Close_CRM", "Less_Annoying_CRM",
    "OnePageCRM", "Capsule_CRM", "amoCRM", "Bitrix24_CRM", "Highrise_CRM",
    "Agile_CRM", "Nimble_CRM", "Streak_for_Gmail", "Copper_CRM"
]

# Define the modules and their associated data fields, including procurement
modules = {
    "Marketing": ["Campaign_ID", "Campaign_Name", "Channel", "Target_Audience", "Spend", "Leads_Generated", "Conversions"],
    "Sales": ["Opportunity_ID", "Customer_ID", "Product", "Quantity", "Sales_Amount", "Close_Date", "Sales_Stage"],
    "Procurement": ["Purchase_Order_ID", "Supplier_ID", "Item", "Quantity", "Unit_Price", "Total_Cost", "Order_Date", "Delivery_Date", "Payment_Terms"],
    "Services": ["Service_Request_ID", "Customer_ID", "Service_Type", "Priority", "Status", "Resolution_Time", "Satisfaction_Rating"],
    "Finance": ["Transaction_ID", "Account_ID", "Amount", "Transaction_Date", "Transaction_Type", "Payment_Method", "Invoice_Number"],
    "HR": ["Employee_ID", "Name", "Department", "Position", "Hire_Date", "Salary", "Performance_Rating"],
    "Inventory": ["Product_ID", "Quantity_on_Hand", "Reorder_Point", "Warehouse_Location", "Unit_Cost", "Shelf_Life", "Batch_Number"],
    "Customer_Service": ["Case_ID", "Customer_ID", "Issue_Type", "Status", "Created_Date", "Closed_Date", "Agent_ID"],
    "Supply_Chain": ["Order_ID", "Supplier_ID", "Product_ID", "Order_Quantity", "Ship_Date", "Delivery_Date", "Shipping_Method"],
    "Manufacturing": ["Work_Order_ID", "Product_ID", "Quantity_Produced", "Start_Date", "End_Date", "Machine_ID", "Defect_Rate"]
}

# Define data types for each field, including procurement
data_types = {
    "Campaign_ID": "int",
    "Campaign_Name": "str",
    "Channel": "str",
    "Target_Audience": "str",
    "Spend": "float",
    "Leads_Generated": "int",
    "Conversions": "int",
    "Opportunity_ID": "int",
    "Customer_ID": "int",
    "Product": "str",
    "Quantity": "int",
    "Sales_Amount": "float",
    "Close_Date": "datetime",
    "Sales_Stage": "str",
    "Purchase_Order_ID": "int",  # Procurement
    "Supplier_ID": "int",      # Procurement
    "Item": "str",             # Procurement
    "Unit_Price": "float",    # Procurement
    "Total_Cost": "float",      # Procurement
    "Order_Date": "datetime",  # Procurement
    "Delivery_Date": "datetime", # Procurement
    "Payment_Terms": "str",    # Procurement
    "Service_Request_ID": "int",
    "Service_Type": "str",
    "Priority": "str",
    "Status": "str",
    "Resolution_Time": "float",
    "Satisfaction_Rating": "int",
    "Transaction_ID": "int",
    "Account_ID": "int",
    "Amount": "float",
    "Transaction_Date": "datetime",
    "Transaction_Type": "str",
    "Payment_Method": "str",
    "Invoice_Number": "str",
    "Employee_ID": "int",
    "Name": "str",
    "Department": "str",
    "Position": "str",
    "Hire_Date": "datetime",
    "Salary": "float",
    "Performance_Rating": "float",
    "Product_ID": "int",
    "Quantity_on_Hand": "int",
    "Reorder_Point": "int",
    "Warehouse_Location": "str",
    "Unit_Cost": "float",
    "Shelf_Life": "int",
    "Batch_Number": "str",
    "Case_ID": "int",
    "Issue_Type": "str",
    "Created_Date": "datetime",
    "Closed_Date": "datetime",
    "Agent_ID": "int",
     "Order_ID": "int",
    "Order_Quantity": "int",
    "Ship_Date": "datetime",
    "Shipping_Method": "str",
    "Work_Order_ID": "int",
    "Quantity_Produced": "int",
    "Start_Date": "datetime",
    "End_Date": "datetime",
    "Machine_ID": "str",
    "Defect_Rate": "float"
}

# Define value ranges or possible values for each field, including procurement
value_ranges = {
    "Campaign_ID": (1000, 9999),
    "Campaign_Name": ["Spring Campaign", "Summer Sale", "Fall Promotion", "Winter Special", "New Product Launch", "Customer Appreciation"],
    "Channel": ["Email", "Social Media", "Website", "Print", "Event", "Telemarketing"],
    "Target_Audience": ["Existing Customers", "Potential Customers", "Partners", "Employees", "General Public"],
    "Spend": (1000, 50000),
    "Leads_Generated": (10, 1000),
    "Conversions": (1, 100),
    "Opportunity_ID": (2000, 8999),
    "Customer_ID": (1001, 5000),
    "Product": ["Product A", "Product B", "Product C", "Product D", "Product E", "Service X", "Service Y"],
    "Quantity": (1, 100),
    "Sales_Amount": (100, 100000),
    "Close_Date": (pd.to_datetime("2023-01-01"), pd.to_datetime("2024-12-31")),
    "Sales_Stage": ["Prospecting", "Qualification", "Proposal", "Negotiation", "Closed Won", "Closed Lost"],
    "Purchase_Order_ID": (3000, 7999),  # Procurement
    "Supplier_ID": (5001, 7000),      # Procurement
    "Item": ["Raw Material 1", "Raw Material 2", "Component A", "Component B", "Packaging Material", "Service Z"], #Procurement
    "Unit_Price": (10, 1000),         # Procurement
    "Total_Cost": (100, 1000000),    # Procurement
    "Order_Date": (pd.to_datetime("2023-01-01"), pd.to_datetime("2024-12-31")),  # Procurement
    "Delivery_Date": (pd.to_datetime("2023-01-15"), pd.to_datetime("2025-01-15")), # Procurement
    "Payment_Terms": ["Net 30", "Net 60", "Net 90", "2% 10 Net 30", "Cash on Delivery"],    # Procurement
    "Service_Request_ID": (4000, 6999),
    "Service_Type": ["Installation", "Maintenance", "Repair", "Consulting", "Training"],
    "Priority": ["High", "Medium", "Low"],
    "Status": ["Open", "In Progress", "Pending", "Resolved", "Closed"],
    "Resolution_Time": (1, 100),
    "Satisfaction_Rating": (1, 5),
    "Transaction_ID": (5000, 5999),
    "Account_ID": (6001, 8000),
    "Amount": (10, 10000),
    "Transaction_Date": (pd.to_datetime("2023-01-01"), pd.to_datetime("2024-12-31")),
    "Transaction_Type": ["Sales", "Purchase", "Expense", "Revenue", "Payment"],
    "Payment_Method": ["Credit Card", "Debit Card", "Cash", "Bank Transfer", "Check"],
    "Invoice_Number": lambda: f"INV-{random.randint(10000, 99999)}",
    "Employee_ID": (7001, 9000),
    "Name": lambda: f"{random.choice(['John', 'Jane', 'Mike', 'Emily'])} {random.choice(['Smith', 'Doe', 'Williams', 'Brown'])}",
    "Department": ["Sales", "Marketing", "Finance", "HR", "Operations", "IT", "R&D"],
    "Position": ["Manager", "Analyst", "Specialist", "Director", "Executive", "Intern"],
    "Hire_Date": (pd.to_datetime("2020-01-01"), pd.to_datetime("2023-12-31")),
    "Salary": (50000, 150000),
    "Performance_Rating": (1, 5),
    "Product_ID": (9001, 9999),
    "Quantity_on_Hand": (0, 1000),
    "Reorder_Point": (10, 200),
    "Warehouse_Location": ["Warehouse A", "Warehouse B", "Warehouse C", "Distribution Center 1"],
    "Unit_Cost": (10, 500),
    "Shelf_Life": (30, 365),
    "Batch_Number": lambda: f"BATCH-{random.randint(100, 999)}",
    "Case_ID": (10001, 15000),
    "Issue_Type": ["Technical Support", "Billing Inquiry", "Product Return", "Service Request", "Complaint"],
    "Created_Date": (pd.to_datetime("2023-01-01"), pd.to_datetime("2024-12-31")),
    "Closed_Date": (pd.to_datetime("2023-01-05"), pd.to_datetime("2025-01-30")),
    "Agent_ID": (2001, 2500),
    "Order_ID": (15001, 20000),
    "Order_Quantity": (1, 200),
    "Ship_Date": (pd.to_datetime("2023-01-01"), pd.to_datetime("2024-12-31")),
    "Shipping_Method": ["Ground", "Air", "Sea", "Express"],
    "Work_Order_ID": (20001, 25000),
    "Quantity_Produced": (1, 1000),
    "Start_Date": (pd.to_datetime("2023-01-01"), pd.to_datetime("2024-12-31")),
    "End_Date": (pd.to_datetime("2023-01-05"), pd.to_datetime("2025-01-30")),
    "Machine_ID": lambda: f"MCH-{random.randint(1, 10)}",
    "Defect_Rate": (0, 0.05)  # 0 to 5%
}

# Define realistic row counts based on system type and typical usage
realistic_row_counts = {
    "SAP_ERP": (10000, 1000000),  # Large enterprises, high volume
    "Salesforce_CRM": (5000, 500000),  # Varied, depends on org size
    "Oracle_EBS": (8000, 800000),
    "Microsoft_Dynamics_365": (6000, 600000),
    "NetSuite_ERP": (3000, 300000),  # Mid-sized businesses
    "HubSpot_CRM": (2000, 200000),  # Smaller businesses, marketing focus
    "Zoho_CRM": (1500, 150000),
    "Infor_ERP": (7000, 700000),
    "Sage_Intacct": (2500, 250000),
    "Workday_ERP": (4000, 400000),  # Focus on HR, but can be large
    "SAP_S4HANA": (12000, 1200000),
    "Salesforce_Sales_Cloud": (6000, 600000),
    "Oracle_Fusion": (9000, 900000),
    "Microsoft_Dynamics_Sales": (5000, 500000),
    "Odoo_ERP_CRM": (2000, 200000),  # Smaller to mid-sized
    "SugarCRM": (3000, 300000),
    "SuiteCRM": (1000, 100000),
    "Epicor_ERP": (4000, 400000),
    "Acumatica_ERP": (2500, 250000),
    "Priority_ERP": (3500, 350000),
    "Teamcenter_PLM": (5000, 200000),  # PLM - related to manufacturing, so potentially large
    "Kinaxis_RapidResponse": (4000, 150000), # Supply Chain
    "Coupa_Procure_to_Pay": (3000, 100000),
    "ServiceNow_CSM": (7000, 600000),
    "Zendesk_Sell": (2000, 150000),
    "Freshsales_CRM": (1500, 100000),
    "Pipedrive_CRM": (1000, 80000),
    "Monday_CRM": (2500, 180000),
    "Vtiger_CRM": (1200, 90000),
    "Insightly_CRM": (1000, 70000),
    "Maximizer_CRM": (800, 60000),
    "Really_Simple_Systems_CRM": (500, 40000),
    "Gold-Vision_CRM": (600, 50000),
    "Raynet_CRM": (700, 55000),
    "Daylite_CRM": (400, 30000),
    "InfoFlo_CRM": (550, 45000),
    "Base_CRM": (900, 75000),
    "Nutshell_CRM": (1100, 85000),
    "Close_CRM": (1300, 95000),
    "Less_Annoying_CRM": (450, 35000),
    "OnePageCRM": (1050, 80000),
    "Capsule_CRM": (1250, 92000),
    "amoCRM": (1400, 105000),
    "Bitrix24_CRM": (2000, 150000),
    "Highrise_CRM": (650, 52000),
    "Agile_CRM": (1150, 88000),
    "Nimble_CRM": (1000, 77000),
    "Streak_for_Gmail": (500, 42000),
     "Copper_CRM": (1200, 90000)
}


def generate_synthetic_data(system_name, num_rows):
    """
    Generates synthetic data for a given ERP/CRM system.

    Args:
        system_name (str): The name of the ERP/CRM system.
        num_rows (int): The number of rows to generate.

    Returns:
        pd.DataFrame: A DataFrame containing the synthetic data, or None if the system is not found.
    """
    # 1. Map system name to modules.  For simplicity, we'll use a basic mapping.
    if system_name in ["SAP_ERP", "Oracle_EBS", "Microsoft_Dynamics_365", "NetSuite_ERP", "Infor_ERP", "Sage_Intacct", "Workday_ERP", "SAP_S4HANA", "Odoo_ERP_CRM", "Epicor_ERP", "Acumatica_ERP", "Priority_ERP"]:
        #  Include Procurement for relevant ERP systems
        relevant_modules = ["Marketing", "Sales", "Procurement", "Services", "Finance", "HR", "Inventory", "Customer_Service", "Supply_Chain", "Manufacturing"]
    elif system_name in ["Salesforce_CRM", "HubSpot_CRM", "Zoho_CRM", "Salesforce_Sales_Cloud", "SugarCRM", "SuiteCRM", "Zendesk_Sell", "Freshsales_CRM", "Pipedrive_CRM", "Monday_CRM", "Vtiger_CRM", "Insightly_CRM",
                           "Maximizer_CRM", "Really_Simple_Systems_CRM", "Gold-Vision_CRM", "Raynet_CRM", "Daylite_CRM", "InfoFlo_CRM", "Base_CRM", "Nutshell_CRM", "Close_CRM", "Less_Annoying_CRM",
                           "OnePageCRM", "Capsule_CRM", "amoCRM", "Bitrix24_CRM", "Highrise_CRM", "Agile_CRM", "Nimble_CRM", "Streak_for_Gmail", "Copper_CRM"]:
        relevant_modules = ["Marketing", "Sales", "Services", "Customer_Service"]  # CRM systems
    elif system_name in ["Teamcenter_PLM", "Kinaxis_RapidResponse", "Coupa_Procure_to_Pay"]:
        relevant_modules = ["Procurement", "Supply_Chain", "Manufacturing"]
    else:
        print(f"System {system_name} not found in our mapping.")
        return None

    # 2.  Collect all fields across the selected modules.
    all_fields = []
    for module in relevant_modules:
        all_fields.extend(modules[module])

    # 3. Create the dataset.
    data = {}
    for field in all_fields:
        if field in value_ranges:
            field_type = data_types[field]
            if field_type == "int":
                low, high = value_ranges[field]
                data[field] = np.random.randint(low, high + 1, num_rows)
            elif field_type == "float":
                low, high = value_ranges[field]
                data[field] = np.random.uniform(low, high, num_rows)
            elif field_type == "str":
                possible_values = value_ranges[field]
                if callable(possible_values):  # Handle lambda functions
                    data[field] = [possible_values() for _ in range(num_rows)]
                else:
                  data[field] = np.random.choice(possible_values, num_rows)
            elif field_type == "datetime":
                start_date, end_date = value_ranges[field]
                data[field] = pd.to_datetime(np.random.uniform(start_date.value, end_date.value, num_rows)).round('D')
        else:
            data[field] = [None] * num_rows  # Handle fields without specific ranges

    df = pd.DataFrame(data)
    return df

# Generate and save the datasets
for i, system_name in enumerate(systems):
    # Use the realistic row counts
    if system_name in realistic_row_counts:
        min_rows, max_rows = realistic_row_counts[system_name]
        num_rows = random.randint(min_rows, max_rows)
    else:
        num_rows = random.randint(100, 1000)  # Default range if not found
    df = generate_synthetic_data(system_name, num_rows)
    if df is not None: # Check if a DataFrame was actually created.
        filename = os.path.join(base_directory, f"{system_name}_synthetic_{i+1}.csv")
        df.to_csv(filename, index=False)
        print(f"Generated and saved dataset: {filename}")
    else:
        print(f"Skipped dataset generation for {system_name} as it was not found.")

print("All datasets generated and saved.")
