# Code Review Report

## Summary

Found 5 issues requiring attention.

---

### 1. SQL injection vulnerability in user lookup
- [x] Skip

**Validation:** valid | **Importance:** HIGH | **File:** src/services/auth_service.py

The user lookup query uses string formatting instead of parameterized queries, which could allow SQL injection attacks.

---

### 2. Missing password hashing
- [ ] Skip

**Validation:** valid | **Importance:** HIGH | **File:** src/services/auth_service.py

The password is stored directly without hashing in the register function.

---

### 3. Rate limiter not applied to registration
- [x] Skip

**Validation:** needs-human-decision | **Importance:** MEDIUM | **File:** src/api/auth_routes.py

The registration endpoint is missing rate limiting.

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
