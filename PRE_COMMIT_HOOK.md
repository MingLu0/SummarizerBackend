# Pre-Commit Hook System

## Overview

This project now enforces the rule: **ALL TESTS MUST PASS BEFORE COMMITTING**. This is implemented through a Git pre-commit hook that automatically runs tests before allowing any commit.

## What It Does

### ✅ Pre-Commit Hook (`.git/hooks/pre-commit`)
- **Automatically runs tests** before every `git commit`
- **Blocks commits** if any tests fail
- **Provides helpful error messages** and guidance
- **Can be bypassed** with `--no-verify` (not recommended)

### ✅ Test Runner Script (`scripts/run-tests.sh`)
- **Comprehensive test suite runner** for manual testing
- **Runs different test categories** separately
- **Provides detailed reporting** and success/failure tracking
- **Helps identify** which specific tests are failing

## How It Works

### 1. Automatic Test Execution
Every time you run `git commit`, the pre-commit hook:
1. Runs core tests (main, middleware, logging, errors)
2. Checks if all tests pass
3. Allows commit if tests pass
4. Blocks commit if tests fail

### 2. Test Categories
- **Core Tests**: Always run in pre-commit hook
  - `test_main.py` - Main application tests
  - `test_middleware.py` - Request middleware tests
  - `test_logging.py` - Logging system tests
  - `test_errors.py` - Error handling tests

- **Full Test Suite**: Available via `scripts/run-tests.sh`
  - All test files including API, services, 502 prevention, etc.

## Usage

### Normal Development Workflow
```bash
# Make your changes
git add .
git commit -m "Your commit message"
# Tests run automatically - commit only succeeds if tests pass
```

### Manual Test Running
```bash
# Run comprehensive test suite
./scripts/run-tests.sh

# Run specific test files
python -m pytest tests/test_main.py -v

# Run all tests
python -m pytest tests/ -v
```

### Emergency Bypass (NOT RECOMMENDED)
```bash
# Only use in emergencies - bypasses all tests
git commit --no-verify -m "Emergency commit"
```

## Benefits

### 🛡️ Quality Assurance
- **Prevents broken code** from being committed
- **Enforces test-driven development**
- **Maintains code quality standards**
- **Prevents regressions** from being introduced

### 🚀 Development Efficiency
- **Catches issues early** before they reach the repository
- **Provides immediate feedback** on code changes
- **Reduces debugging time** in production
- **Maintains consistent code quality**

### 📊 Team Collaboration
- **Ensures all team members** follow the same standards
- **Prevents "it works on my machine"** issues
- **Maintains repository stability**
- **Facilitates code reviews**

## Configuration

### Pre-Commit Hook Location
- **File**: `.git/hooks/pre-commit`
- **Permissions**: Executable (`chmod +x`)
- **Scope**: Project-specific (not shared via git)

### Test Runner Location
- **File**: `scripts/run-tests.sh`
- **Permissions**: Executable (`chmod +x`)
- **Scope**: Project-wide (shared via git)

## Troubleshooting

### Tests Fail in Pre-Commit Hook
1. **Fix the failing tests** first
2. **Run tests manually** to verify: `python -m pytest tests/ -v`
3. **Try committing again**

### Pre-Commit Hook Not Working
1. **Check permissions**: `ls -la .git/hooks/pre-commit`
2. **Make executable**: `chmod +x .git/hooks/pre-commit`
3. **Verify content**: `cat .git/hooks/pre-commit`

### Need to Disable Temporarily
```bash
# Rename the hook to disable it
mv .git/hooks/pre-commit .git/hooks/pre-commit.disabled

# Rename back to enable it
mv .git/hooks/pre-commit.disabled .git/hooks/pre-commit
```

## Best Practices

### ✅ Do
- **Write tests** for all new features
- **Fix failing tests** before committing
- **Run full test suite** before major changes
- **Use descriptive commit messages**

### ❌ Don't
- **Bypass tests** with `--no-verify` unless absolutely necessary
- **Commit broken code** even temporarily
- **Ignore test failures** or warnings
- **Skip writing tests** for new functionality

## Success Metrics

- ✅ **Zero broken commits** in repository history
- ✅ **All tests pass** before every commit
- ✅ **Consistent code quality** across all changes
- ✅ **Reduced production bugs** due to early detection
- ✅ **Faster development cycles** with immediate feedback

---

**Remember**: The pre-commit hook is your friend! It helps maintain code quality and prevents issues from reaching production. Embrace it as part of your development workflow.
