import json
from pathlib import Path
from processlens.models.process import ProcessModel, Step

DATA_DOCS_DIR = Path("data/documents")
DATA_GOLD_DIR = Path("data/gold_labels")

DATA_DOCS_DIR.mkdir(parents=True, exist_ok=True)
DATA_GOLD_DIR.mkdir(parents=True, exist_ok=True)

DATASET_SPECS = [
    {
        "id": "sop_01_invoice_approval",
        "title": "Standard Operating Procedure: Accounts Payable Invoice Approval",
        "actors": ["Vendor", "Accounts Payable Clerk", "Department Manager", "Finance Director"],
        "text": """STANDARD OPERATING PROCEDURE: ACCOUNTS PAYABLE INVOICE APPROVAL
DOCUMENT ID: SOP-FIN-001
VERSION: 2.1
EFFECTIVE DATE: OCTOBER 2024

1. PURPOSE & OBJECTIVE
This Standard Operating Procedure defines the end-to-end workflow for receiving, validating, authorizing, and disbursing payments for commercial vendor invoices. The goal is to enforce budgetary discipline, prevent duplicate payments, and maintain compliance with internal financial controls.

2. SCOPE & RESPONSIBLE ACTORS
This procedure applies to all operational expenditure invoices received across company business units.
- Vendor: External supplier delivering goods/services and generating billing.
- Accounts Payable Clerk: Initial receiver responsible for header validation and two-way match.
- Department Manager: Budget owner validating operational receipt of deliverables.
- Finance Director: Executive officer authorizing high-value capital or operational disbursements.

3. PROCEDURAL WORKFLOW

[Chunk 0: Invoice Intake and Header Verification]
The process initiates when the Vendor transmits an invoice via the digital invoicing portal. Upon receipt, the Accounts Payable Clerk logs the invoice into the ERP system, verifying vendor identification number, tax registration, and payment terms against the master vendor file.

[Chunk 1: Two-Way Match Verification]
The Accounts Payable Clerk executes a two-way match comparing invoice line items, unit quantities, and total dollar amounts against the approved purchase order. If a discrepancy greater than 1% exists, the invoice is flagged for vendor clarification.

[Chunk 2: Departmental Budget Review]
Following two-way match validation, the Accounts Payable Clerk routes the invoice to the Department Manager. The Department Manager reviews the invoice against the departmental operational budget and confirms satisfactory receipt of goods or services.

[Chunk 3: Manager Approval Decision]
The Department Manager makes an authorization decision:
- If rejected or disputed, the invoice is returned to the Accounts Payable Clerk with comments to request credit notes or corrections from the vendor.
- If approved, the system evaluates the invoice total amount:
  * Invoices under $10,000 proceed directly to payment scheduling.
  * Invoices equal to or exceeding $10,000 require secondary executive approval by the Finance Director.

[Chunk 4: Executive Finance Authorization]
For invoices requiring secondary approval, the Finance Director audits contract terms, executive budget allocation, and cash-flow impact. The Finance Director signs off on the authorization ledger.

[Chunk 5: Payment Disbursement & Reconciliation]
Once fully approved, the Accounts Payable Clerk queues the voucher for electronic funds transfer (EFT) batch execution, triggers remittance notifications to the Vendor, and archives the transaction in the general ledger.""",
        "steps": [
            Step(
                id="step_1",
                name="Submit and log invoice",
                description="Vendor submits invoice through portal and AP Clerk logs into ERP",
                actor="Accounts Payable Clerk",
                inputs=["Vendor invoice", "Master vendor file"],
                outputs=["Logged ERP voucher"],
                next_step_ids=["step_2"],
                is_decision=False,
                conditions=[],
                source_chunk_ids=["chunk_0"],
                confidence=0.98,
                is_loop=False,
            ),
            Step(
                id="step_2",
                name="Perform two-way match",
                description="AP Clerk compares invoice against purchase order and line items",
                actor="Accounts Payable Clerk",
                inputs=["Logged ERP voucher", "Purchase order"],
                outputs=["Match verification report"],
                next_step_ids=["step_3"],
                is_decision=False,
                conditions=[],
                source_chunk_ids=["chunk_1"],
                confidence=0.96,
                is_loop=False,
            ),
            Step(
                id="step_3",
                name="Departmental operational review",
                description="Department Manager reviews service delivery and budget allocation",
                actor="Department Manager",
                inputs=["Match verification report", "Department budget"],
                outputs=["Manager review assessment"],
                next_step_ids=["step_4"],
                is_decision=False,
                conditions=[],
                source_chunk_ids=["chunk_2"],
                confidence=0.95,
                is_loop=False,
            ),
            Step(
                id="step_4",
                name="Manager authorization decision",
                description="Department Manager decides whether to approve or return invoice",
                actor="Department Manager",
                inputs=["Manager review assessment"],
                outputs=["Approval decision"],
                next_step_ids=["step_1", "step_5", "step_6"],
                is_decision=True,
                conditions=["Disputed / Needs correction", "Approved under $10,000", "Approved >= $10,000"],
                source_chunk_ids=["chunk_3"],
                confidence=0.94,
                is_loop=True,
            ),
            Step(
                id="step_5",
                name="Executive finance sign-off",
                description="Finance Director authorizes high-value invoices exceeding threshold",
                actor="Finance Director",
                inputs=["Approval decision", "Executive ledger"],
                outputs=["Executive authorization stamp"],
                next_step_ids=["step_6"],
                is_decision=False,
                conditions=[],
                source_chunk_ids=["chunk_4"],
                confidence=0.97,
                is_loop=False,
            ),
            Step(
                id="step_6",
                name="Disburse payment and reconcile",
                description="AP Clerk schedules EFT payment and posts journal entry",
                actor="Accounts Payable Clerk",
                inputs=["Approved voucher"],
                outputs=["Payment remittance advice", "Reconciled ledger"],
                next_step_ids=[],
                is_decision=False,
                conditions=[],
                source_chunk_ids=["chunk_5"],
                confidence=0.99,
                is_loop=False,
            ),
        ],
    },
    {
        "id": "sop_02_employee_onboarding",
        "title": "Standard Operating Procedure: Corporate Employee Onboarding",
        "actors": ["HR Specialist", "Hiring Manager", "IT Support Engineer", "New Employee"],
        "text": """STANDARD OPERATING PROCEDURE: CORPORATE EMPLOYEE ONBOARDING
DOCUMENT ID: SOP-HR-002
VERSION: 1.4
EFFECTIVE DATE: NOVEMBER 2024

1. PURPOSE & SCOPE
Ensures a structured, secure, and welcoming transition for newly hired personnel into corporate operations.

2. ROLES
- HR Specialist: Coordinates paperwork, background verification, and corporate benefits orientation.
- IT Support Engineer: Provisions hardware, federated credentials, and SaaS licensing.
- Hiring Manager: Establishes 30-60-90 day performance goals and team welcome.
- New Employee: Completes compliance modules, profile setup, and policy acknowledgments.

3. WORKFLOW STEPS

[Chunk 0: Pre-Arrival Verification & Welcome Packet]
Upon signed offer acceptance, the HR Specialist initiates identity checks, reference verification, and criminal background screening. Once cleared, HR sends the digital welcome packet to the candidate.

[Chunk 1: IT Asset and Access Provisioning]
One week prior to start date, the IT Support Engineer provisions a corporate workstation, configures multi-factor authentication (MFA), creates domain credentials, and assigns departmental role permissions.

[Chunk 2: Day One Orientation]
On the first working day, the HR Specialist conducts an official orientation session covering corporate culture, compliance regulations, health insurance, and standard workplace policies.

[Chunk 3: Team Integration & Workspace Setup]
The Hiring Manager welcomes the New Employee, introduces team colleagues, assigns a peer mentor, and reviews immediate project deliverables and expectations.

[Chunk 4: Policy Acknowledgment and 30-Day Check-in]
The New Employee completes mandated information security training and submits signed acknowledgment forms. At day 30, the Hiring Manager conducts a formal onboarding assessment.""",
        "steps": [
            Step(
                id="step_1",
                name="Background verification and welcome packet",
                description="HR verifies candidate background and dispatches welcome packet",
                actor="HR Specialist",
                inputs=["Signed offer letter"],
                outputs=["Cleared background check", "Dispatched packet"],
                next_step_ids=["step_2"],
                is_decision=False,
                conditions=[],
                source_chunk_ids=["chunk_0"],
                confidence=0.98,
                is_loop=False,
            ),
            Step(
                id="step_2",
                name="Hardware and access provisioning",
                description="IT provisions laptop, domain account, and MFA credentials",
                actor="IT Support Engineer",
                inputs=["HR onboarding ticket"],
                outputs=["Configured laptop", "Active SSO account"],
                next_step_ids=["step_3"],
                is_decision=False,
                conditions=[],
                source_chunk_ids=["chunk_1"],
                confidence=0.97,
                is_loop=False,
            ),
            Step(
                id="step_3",
                name="Conduct Day One HR orientation",
                description="HR leads company overview and benefits enrollment",
                actor="HR Specialist",
                inputs=["Employee presence", "Benefits documentation"],
                outputs=["Completed orientation signoff"],
                next_step_ids=["step_4"],
                is_decision=False,
                conditions=[],
                source_chunk_ids=["chunk_2"],
                confidence=0.99,
                is_loop=False,
            ),
            Step(
                id="step_4",
                name="Team introduction and 30-day goal alignment",
                description="Hiring manager introduces team, sets deliverables and mentor",
                actor="Hiring Manager",
                inputs=["Department charter", "Role objectives"],
                outputs=["Signed 30-day performance plan"],
                next_step_ids=["step_5"],
                is_decision=False,
                conditions=[],
                source_chunk_ids=["chunk_3"],
                confidence=0.95,
                is_loop=False,
            ),
            Step(
                id="step_5",
                name="Complete mandatory training and 30-day review",
                description="Employee finishes compliance learning and manager evaluates onboarding",
                actor="New Employee",
                inputs=["Compliance LMS modules"],
                outputs=["Training certification", "Finalized onboarding record"],
                next_step_ids=[],
                is_decision=False,
                conditions=[],
                source_chunk_ids=["chunk_4"],
                confidence=0.96,
                is_loop=False,
            ),
        ],
    },
    {
        "id": "sop_03_customer_refund",
        "title": "Standard Operating Procedure: Customer Refund & Credit Processing",
        "actors": ["Customer", "Support Agent", "Billing Specialist", "Finance Manager"],
        "text": """STANDARD OPERATING PROCEDURE: CUSTOMER REFUND PROCESSING
DOCUMENT ID: SOP-OPS-003
VERSION: 3.0

1. PURPOSE
Provides standard criteria for evaluating customer refund requests, preventing fraud, and executing timely reimbursements.

2. ACTORS
- Customer: Submits return/refund request via portal.
- Support Agent: Evaluates request eligibility against policy timeline.
- Billing Specialist: Validates payment token and initiates transaction reversal.
- Finance Manager: Approves high-value refunds exceeding $500.

3. PROCEDURE

[Chunk 0: Refund Request Intake]
The Customer submits a refund request through the self-service web portal citing reason for return, order reference number, and item receipt.

[Chunk 1: Eligibility and Order Audit]
The Support Agent reviews order timestamp and return policy terms.
- If purchase was made within 30 calendar days and product return conditions are satisfied, request is approved.
- If request is outside 30 days or ineligible, Support Agent issues policy rejection explanation to Customer.

[Chunk 2: Value Threshold Assessment]
For eligible requests, the Support Agent determines total refund amount:
- If refund is $500 or less, request proceeds directly to payment reversal.
- If refund exceeds $500, request is escalated to Finance Manager for review.

[Chunk 3: Manager Authorization]
The Finance Manager inspects customer transaction history, fraud scores, and merchant processor logs before signing approval.

[Chunk 4: Gateway Reversal and Notification]
The Billing Specialist issues refund via payment gateway token. Support Agent transmits transaction confirmation receipt to Customer.""",
        "steps": [
            Step(
                id="step_1",
                name="Submit refund request",
                description="Customer lodges request with order ID and justification",
                actor="Customer",
                inputs=["Order receipt", "Return reason"],
                outputs=["Support ticket"],
                next_step_ids=["step_2"],
                is_decision=False,
                conditions=[],
                source_chunk_ids=["chunk_0"],
                confidence=0.99,
                is_loop=False,
            ),
            Step(
                id="step_2",
                name="Evaluate 30-day policy eligibility",
                description="Support Agent checks purchase age and product return conditions",
                actor="Support Agent",
                inputs=["Support ticket", "Return policy"],
                outputs=["Eligibility determination"],
                next_step_ids=["step_3", "step_5"],
                is_decision=True,
                conditions=["Eligible within policy", "Ineligible / Out of policy"],
                source_chunk_ids=["chunk_1"],
                confidence=0.97,
                is_loop=False,
            ),
            Step(
                id="step_3",
                name="Evaluate refund amount threshold",
                description="Check if refund exceeds $500 threshold requiring manager sign-off",
                actor="Support Agent",
                inputs=["Eligibility determination", "Order total"],
                outputs=["Threshold route"],
                next_step_ids=["step_4", "step_6"],
                is_decision=True,
                conditions=["Amount > $500", "Amount <= $500"],
                source_chunk_ids=["chunk_2"],
                confidence=0.96,
                is_loop=False,
            ),
            Step(
                id="step_4",
                name="Manager fraud audit and sign-off",
                description="Finance Manager reviews fraud score and approves high-value refund",
                actor="Finance Manager",
                inputs=["Threshold route", "Customer history"],
                outputs=["Manager authorization"],
                next_step_ids=["step_6"],
                is_decision=False,
                conditions=[],
                source_chunk_ids=["chunk_3"],
                confidence=0.95,
                is_loop=False,
            ),
            Step(
                id="step_5",
                name="Send rejection notice",
                description="Support Agent closes ticket and sends policy explanation to customer",
                actor="Support Agent",
                inputs=["Eligibility determination"],
                outputs=["Rejection notice email"],
                next_step_ids=[],
                is_decision=False,
                conditions=[],
                source_chunk_ids=["chunk_1"],
                confidence=0.98,
                is_loop=False,
            ),
            Step(
                id="step_6",
                name="Process payment gateway reversal",
                description="Billing Specialist executes reversal and issues credit notification",
                actor="Billing Specialist",
                inputs=["Manager authorization or standard approved ticket"],
                outputs=["Gateway transaction receipt", "Customer confirmation"],
                next_step_ids=[],
                is_decision=False,
                conditions=[],
                source_chunk_ids=["chunk_4"],
                confidence=0.99,
                is_loop=False,
            ),
        ],
    },
    {
        "id": "sop_04_security_incident_response",
        "title": "Standard Operating Procedure: Cybersecurity Incident Triage & Containment",
        "actors": ["SOC Analyst", "Incident Commander", "Security Engineer", "Communications Officer"],
        "text": """STANDARD OPERATING PROCEDURE: CYBERSECURITY INCIDENT RESPONSE
DOCUMENT ID: SOP-SEC-004
VERSION: 4.2

1. PURPOSE & SCOPE
Defines rapid response protocol for cyber threats, intrusions, and security anomalies.

2. ROLES
- SOC Analyst: Triage alerts from SIEM, EDR, and intrusion detection systems.
- Incident Commander: Directs containment strategy, escalations, and resource allocation.
- Security Engineer: Executes technical isolation, packet capture, and malware remediation.
- Communications Officer: Manages stakeholder briefings and regulatory notifications.

3. PROCEDURE

[Chunk 0: Alert Detection and Initial Triage]
The SOC Analyst detects anomalous behavior via SIEM automated alarms. The Analyst investigates indicators of compromise (IoC) and cross-references threat intelligence feeds.

[Chunk 1: Severity Classification]
The SOC Analyst classifies the event:
- P3/P4 Low-impact: Handled via standard automated host containment.
- P1/P2 Critical: Escalated directly to Incident Commander within 15 minutes.

[Chunk 2: Incident Command Activation]
The Incident Commander declares an active security incident, opens a dedicated war room, and assigns technical response teams.

[Chunk 3: Host Isolation and Containment]
The Security Engineer isolates affected subnetworks, revokes compromised credentials, and takes memory forensic snapshots before applying containment scripts.

[Chunk 4: Verification and Eradication]
The Security Engineer removes persistent backdoors and patches vulnerabilities. If threat re-emerges during observation, team loops back to containment.

[Chunk 5: Executive Communication & Post-Mortem]
The Communications Officer briefs executive leadership. The Incident Commander compiles the root cause analysis post-mortem report.""",
        "steps": [
            Step(
                id="step_1",
                name="Detect and triage security alert",
                description="SOC Analyst examines SIEM triggers and validates indicators of compromise",
                actor="SOC Analyst",
                inputs=["SIEM alert", "Threat intelligence feed"],
                outputs=["Triage assessment report"],
                next_step_ids=["step_2"],
                is_decision=False,
                conditions=[],
                source_chunk_ids=["chunk_0"],
                confidence=0.98,
                is_loop=False,
            ),
            Step(
                id="step_2",
                name="Severity classification decision",
                description="Determine whether incident is P1/P2 critical or routine P3/P4",
                actor="SOC Analyst",
                inputs=["Triage assessment report"],
                outputs=["Severity grade"],
                next_step_ids=["step_3", "step_4"],
                is_decision=True,
                conditions=["P1/P2 Critical Incident", "P3/P4 Standard Alert"],
                source_chunk_ids=["chunk_1"],
                confidence=0.96,
                is_loop=False,
            ),
            Step(
                id="step_3",
                name="Activate war room and incident command",
                description="Incident Commander mobilizes war room and coordinates technical teams",
                actor="Incident Commander",
                inputs=["P1/P2 Severity grade"],
                outputs=["Active incident war room"],
                next_step_ids=["step_4"],
                is_decision=False,
                conditions=[],
                source_chunk_ids=["chunk_2"],
                confidence=0.97,
                is_loop=False,
            ),
            Step(
                id="step_4",
                name="Isolate affected assets and contain",
                description="Security Engineer isolates compromised endpoints and captures memory dumps",
                actor="Security Engineer",
                inputs=["Asset identifiers", "Containment protocol"],
                outputs=["Isolated subnet", "Forensic snapshots"],
                next_step_ids=["step_5"],
                is_decision=False,
                conditions=[],
                source_chunk_ids=["chunk_3"],
                confidence=0.99,
                is_loop=False,
            ),
            Step(
                id="step_5",
                name="Eradicate threat and verify containment",
                description="Security Engineer removes malware and validates no re-infection",
                actor="Security Engineer",
                inputs=["Forensic snapshots", "Patch definitions"],
                outputs=["Cleaned infrastructure"],
                next_step_ids=["step_4", "step_6"],
                is_decision=True,
                conditions=["Threat persists (loop back)", "Clean verified"],
                source_chunk_ids=["chunk_4"],
                confidence=0.94,
                is_loop=True,
            ),
            Step(
                id="step_6",
                name="Publish post-mortem and stakeholder brief",
                description="Communications Officer and Commander publish root cause report",
                actor="Communications Officer",
                inputs=["Incident timeline", "Forensic root cause"],
                outputs=["Executive incident report", "Post-mortem review"],
                next_step_ids=[],
                is_decision=False,
                conditions=[],
                source_chunk_ids=["chunk_5"],
                confidence=0.98,
                is_loop=False,
            ),
        ],
    },
    {
        "id": "sop_05_it_hardware_provisioning",
        "title": "Standard Operating Procedure: Enterprise IT Hardware Provisioning",
        "actors": ["Employee", "IT Asset Manager", "Systems Engineer", "Logistics Courier"],
        "text": """STANDARD OPERATING PROCEDURE: IT HARDWARE PROVISIONING
DOCUMENT ID: SOP-IT-005
VERSION: 1.8

1. PURPOSE
Specifies standard process for procuring, configuring, encrypting, and dispatching computing devices.

2. ACTORS
- Employee: Submits hardware requisition request.
- IT Asset Manager: Verifies device warranty, inventory availability, and manager approval.
- Systems Engineer: Implements disk encryption, golden image installation, and software packages.
- Logistics Courier: Handles tracked shipping and chain-of-custody delivery.

3. WORKFLOW

[Chunk 0: Requisition Intake]
The Employee submits a hardware request ticket specifying device tier and business justification.

[Chunk 1: Inventory Check and Allocation]
The IT Asset Manager checks inventory repository:
- If required device is in stock, it is allocated and checked out in CMDB.
- If out of stock, purchase order is placed with primary vendor.

[Chunk 2: Imaging and Security Hardening]
The Systems Engineer deploys enterprise golden OS image, activates BitLocker full-disk encryption, registers device in MDM, and installs security endpoint agents.

[Chunk 3: Courier Dispatch & Delivery Confirmation]
The Logistics Courier collects packaged device, delivers to employee address with signature verification, and updates CMDB asset tracking status to active.""",
        "steps": [
            Step(
                id="step_1",
                name="Submit hardware request",
                description="Employee files requisition ticket for computing equipment",
                actor="Employee",
                inputs=["Business justification"],
                outputs=["Hardware ticket"],
                next_step_ids=["step_2"],
                is_decision=False,
                conditions=[],
                source_chunk_ids=["chunk_0"],
                confidence=0.99,
                is_loop=False,
            ),
            Step(
                id="step_2",
                name="Verify inventory availability",
                description="IT Asset Manager checks stock status and assigns asset",
                actor="IT Asset Manager",
                inputs=["Hardware ticket", "CMDB inventory database"],
                outputs=["Assigned hardware unit"],
                next_step_ids=["step_3"],
                is_decision=False,
                conditions=[],
                source_chunk_ids=["chunk_1"],
                confidence=0.97,
                is_loop=False,
            ),
            Step(
                id="step_3",
                name="Deploy image and encrypt device",
                description="Systems Engineer installs golden OS build, MDM enrollment, and BitLocker",
                actor="Systems Engineer",
                inputs=["Assigned hardware unit", "Golden image profile"],
                outputs=["Configured encrypted device"],
                next_step_ids=["step_4"],
                is_decision=False,
                conditions=[],
                source_chunk_ids=["chunk_2"],
                confidence=0.98,
                is_loop=False,
            ),
            Step(
                id="step_4",
                name="Ship and confirm delivery",
                description="Courier delivers device and secures employee signature confirmation",
                actor="Logistics Courier",
                inputs=["Configured encrypted device", "Shipping address"],
                outputs=["Signed delivery receipt", "Activated CMDB record"],
                next_step_ids=[],
                is_decision=False,
                conditions=[],
                source_chunk_ids=["chunk_3"],
                confidence=0.96,
                is_loop=False,
            ),
        ],
    },
]

for i in range(6, 26):
    specs = [
        ("sop_06_vendor_procurement", "Vendor Procurement & Contract Award", ["Procurement Specialist", "Legal Counsel", "Finance Controller", "Supplier"], "Procurement"),
        ("sop_07_customer_kyc_verification", "Customer Know-Your-Customer Verification", ["Customer", "Compliance Officer", "Risk Analyst", "Senior Auditor"], "Compliance"),
        ("sop_08_change_management", "IT Infrastructure Change Management", ["Change Requester", "CAB Chair", "Release Engineer", "Site Reliability Engineer"], "Operations"),
        ("sop_09_employee_offboarding", "Employee Separation and Offboarding", ["Departing Staff", "HR Coordinator", "IT Security Lead", "Payroll Specialist"], "HR"),
        ("sop_10_loan_application_review", "Commercial Loan Application Review", ["Borrower", "Loan Officer", "Credit Underwriter", "Credit Committee"], "Finance"),
        ("sop_11_software_release_deployment", "Automated Software Release & Canary Deployment", ["Release Manager", "QA Engineer", "DevOps Engineer", "SRE"], "Engineering"),
        ("sop_12_patient_intake_hospital", "Emergency Patient Intake & Triage", ["Patient", "Triage Nurse", "Admissions Clerk", "Attending Physician"], "Healthcare"),
        ("sop_13_access_permission_request", "Privileged System Access Request", ["Requestor", "Direct Manager", "Information Security Officer", "System Admin"], "Security"),
        ("sop_14_product_return_and_restocking", "Warehouse Product Return & Restocking", ["Customer", "Receiving Clerk", "QA Inspector", "Inventory Manager"], "Logistics"),
        ("sop_15_travel_expense_reimbursement", "Business Travel Expense Reimbursement", ["Employee", "Line Manager", "Finance Auditor", "Disbursement Clerk"], "Finance"),
        ("sop_16_legal_contract_review", "Commercial Legal Contract Review", ["Business Sponsor", "Legal Counsel", "Counterparty", "General Counsel"], "Legal"),
        ("sop_17_disaster_recovery_failover", "Data Center Disaster Recovery Failover", ["DR Coordinator", "Infrastructure Lead", "Database Administrator", "QA Validator"], "Infrastructure"),
        ("sop_18_customer_churn_retention", "High-Value Customer Churn Intervention", ["Customer Success Manager", "Account Executive", "Product Specialist", "Customer"], "Sales"),
        ("sop_19_cloud_infrastructure_alert", "Cloud Infrastructure Automated Alert Remediation", ["Cloud Monitoring Engine", "On-Call Engineer", "Platform Lead", "Security Auditor"], "Cloud"),
        ("sop_20_payroll_processing", "Monthly Corporate Payroll Execution", ["Payroll Officer", "HR Director", "Treasury Lead", "Internal Auditor"], "HR"),
        ("sop_21_quarterly_inventory_audit", "Quarterly Physical Inventory Cycle Count", ["Audit Lead", "Warehouse Supervisor", "Finance Controller", "Operations VP"], "SupplyChain"),
        ("sop_22_gdpr_data_deletion_request", "GDPR Right-to-Erasure Request Fulfillment", ["Data Subject", "Privacy Officer", "Data Engineer", "Legal Representative"], "Privacy"),
        ("sop_23_marketing_campaign_launch", "Multi-Channel Marketing Campaign Approval", ["Marketing Lead", "Creative Director", "Compliance Reviewer", "Channel Manager"], "Marketing"),
        ("sop_24_quality_assurance_defect_triage", "Software Defect Triage & Hotfix Release", ["QA Tester", "Product Owner", "Software Engineer", "Release Lead"], "Software"),
        ("sop_25_customer_complaint_escalation", "Executive Customer Complaint Resolution", ["Support Specialist", "Tier 2 Lead", "Operations Director", "Customer Relations VP"], "Support"),
    ]
    sop_id, title, actors, domain = specs[i - 6]
    doc_text = f"""STANDARD OPERATING PROCEDURE: {title.upper()}
DOCUMENT ID: SOP-{domain.upper()}-{i:03d}
VERSION: 1.{i%5 + 1}
EFFECTIVE DATE: DECEMBER 2024

1. PURPOSE & PRINCIPLES
Standardizes operations, compliance checkpoints, and handoffs across organizational tiers for {title.lower()}.

2. PARTICIPATING ACTORS
- {actors[0]}: Initiates trigger event and provides source context.
- {actors[1]}: Reviews inputs against regulatory and operational standards.
- {actors[2]}: Performs primary execution, mitigation, or analytical assessment.
- {actors[3]}: Validates resolution sign-off and records audit trails.

3. OPERATIONAL PHASES

[Chunk 0: Intake & Validation]
The process starts when {actors[0]} prepares and submits the standardized package into the operations management system. {actors[1]} verifies submission completeness and logs the transaction.

[Chunk 1: Assessment and Standards Compliance]
{actors[1]} evaluates the request against current organizational risk guidelines, service level commitments, and standard criteria.

[Chunk 2: Governance Decision Gate]
{actors[1]} renders an approval determination:
- If adjustments are required or parameters fail baseline criteria, the request returns to {actors[0]} for revisions.
- If criteria are met, the request advances to {actors[2]} for technical processing.

[Chunk 3: Execution and Verification]
{actors[2]} executes the planned operations, tests operational integrity, and compiles supporting artifacts into the master tracking file.

[Chunk 4: Final Sign-off & Audit Closure]
{actors[3]} reviews execution outcomes, grants final authorization, updates enterprise registers, and archives the case documentation."""

    steps = [
        Step(
            id="step_1",
            name=f"Initiate {title.lower()} intake",
            description=f"{actors[0]} submits initial case payload and {actors[1]} logs ticket",
            actor=actors[0],
            inputs=["Standard intake form"],
            outputs=["Registered case record"],
            next_step_ids=["step_2"],
            is_decision=False,
            conditions=[],
            source_chunk_ids=["chunk_0"],
            confidence=0.98,
            is_loop=False,
        ),
        Step(
            id="step_2",
            name="Evaluate baseline compliance",
            description=f"{actors[1]} reviews submission against organizational guidelines",
            actor=actors[1],
            inputs=["Registered case record"],
            outputs=["Compliance assessment"],
            next_step_ids=["step_3"],
            is_decision=False,
            conditions=[],
            source_chunk_ids=["chunk_1"],
            confidence=0.97,
            is_loop=False,
        ),
        Step(
            id="step_3",
            name="Governance approval decision",
            description=f"{actors[1]} determines whether to accept request or return for revision",
            actor=actors[1],
            inputs=["Compliance assessment"],
            outputs=["Approval decision"],
            next_step_ids=["step_1", "step_4"],
            is_decision=True,
            conditions=["Needs revision / Return", "Approved for processing"],
            source_chunk_ids=["chunk_2"],
            confidence=0.95,
            is_loop=True,
        ),
        Step(
            id="step_4",
            name="Execute core operations",
            description=f"{actors[2]} executes required actions and produces verification report",
            actor=actors[2],
            inputs=["Approval decision"],
            outputs=["Operational execution proof"],
            next_step_ids=["step_5"],
            is_decision=False,
            conditions=[],
            source_chunk_ids=["chunk_3"],
            confidence=0.96,
            is_loop=False,
        ),
        Step(
            id="step_5",
            name="Authorize final sign-off and archive",
            description=f"{actors[3]} certifies completion and archives permanent records",
            actor=actors[3],
            inputs=["Operational execution proof"],
            outputs=["Certified archive record"],
            next_step_ids=[],
            is_decision=False,
            conditions=[],
            source_chunk_ids=["chunk_4"],
            confidence=0.99,
            is_loop=False,
        ),
    ]

    DATASET_SPECS.append({
        "id": sop_id,
        "title": title,
        "actors": actors,
        "text": doc_text,
        "steps": steps,
    })

for item in DATASET_SPECS:
    doc_path = DATA_DOCS_DIR / f"{item['id']}.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(item["text"])

    model = ProcessModel(
        id=item["id"],
        title=item["title"],
        version="1.0",
        actors=item["actors"],
        steps=item["steps"],
        metadata={
            "review_status": "NEEDS_MANUAL_REVIEW",
            "reviewer": None,
            "notes": "Awaiting manual verification by human reviewer",
            "domain": item["id"].split("_")[2],
        },
    )

    gold_path = DATA_GOLD_DIR / f"{item['id']}.json"
    with open(gold_path, "w", encoding="utf-8") as f:
        f.write(model.model_dump_json(indent=2))

print(f"Successfully generated {len(DATASET_SPECS)} documents and gold labels.")
