# Code Review Report

## Summary

Found 5 issues requiring attention.

---

### 1. SQL injection vulnerability in user lookup
- [ ] Skip

**Validation:** valid | **Importance:** HIGH | **File:** src/services/auth_service.py

EDITED BY USER: The user lookup query uses string formatting. This needs to be fixed using parameterized queries with the ? placeholder syntax for SQLite, not %s for PostgreSQL. Make sure to also update the related queries in get_user_by_id function.

---

### 2. Missing password hashing
- [ ] Skip

**Validation:** valid | **Importance:** HIGH | **File:** src/services/auth_service.py

The password is stored directly without hashing in the register function.

---

### 3. Rate limiter not applied to registration
- [ ] Skip

**Validation:** needs-human-decision | **Importance:** MEDIUM | **File:** src/api/auth_routes.py

EDITED BY USER: Use Redis-based rate limiting with a sliding window algorithm. Configure the limit to 5 requests per minute per IP address. Add the rate_limit decorator from our custom middleware.

---

### 4. Hardcoded JWT secret
- [ ] Skip

**Validation:** validation_failed | **Importance:** HIGH | **File:** src/services/auth_service.py

The JWT secret is hardcoded in the source code.

---

### 5. Session cleanup not implemented
- [ ] Skip

**Validation:** invalid | **Importance:** LOW | **File:** src/services/session_service.py

Expired sessions are never cleaned up from the database.

---
