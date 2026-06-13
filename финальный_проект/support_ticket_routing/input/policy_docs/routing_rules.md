# Routing rules for support tickets

This document defines the standard routing logic for customer support tickets.

Every incoming ticket must be assigned to exactly one normalized category and one responsible department. The routing decision must be based primarily on the issue description, not on the raw category from the source dataset, because raw categories may be noisy or inconsistent.

Supported normalized categories:

1. account_access
Use this category for login failures, password problems, authentication issues, two-factor authentication codes, missing verification codes, account lockouts, credential problems, and account access failures.

Responsible department: support_l2.

2. billing_refund
Use this category for failed payments, duplicate charges, refunds, billing statement discrepancies, invoice issues, payment gateway timeouts, transaction failures, and payment method problems.

Responsible department: billing_team.

3. technical_bug
Use this category for crashes, errors, data synchronization failures, broken features, file upload problems, performance issues, report generation bugs, and problems after software updates.

Responsible department: technical_team.

4. subscription
Use this category for subscription cancellation, unexpected renewal, trial issues, plan upgrade, plan downgrade, unclear subscription status, and requests related to billing plans.

Responsible department: support_l2.

If the issue description does not clearly match any supported category, assign the category other and route the ticket to support_l1.

Routing must include a short explanation and an evidence quote from the original ticket. The evidence quote must be copied from the issue description exactly and must not be invented.
