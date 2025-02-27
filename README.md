# Git Branching Strategy and Branch Protection (GitHub Flow)

This document outlines the branching workflow and repository rules for our application repository. We adopt **GitHub Flow** as our branching strategy, which is well-suited for a small team (up to 5 developers) and continuous deployment. The guidelines below cover how we manage branches, enforce protections on the `main` branch, integrate CI/CD with GitHub Actions and Azure, and follow best practices for collaboration.

## Branching Strategy: GitHub Flow

In this workflow, the `main` branch is always kept in a deployable state, and all new development is done in isolated feature branches. Each feature branch is short-lived and merged into `main` via a pull request (PR) after review and testing, then deleted upon merge. This ensures the `main` code remains stable and ready for release at all times, supporting continuous integration and delivery.

Key points of our GitHub Flow branching strategy:

- **`main` Branch** – The default branch is **`main`**, which always contains tested, deployment-ready code. We treat `main` as the source of truth for production; no broken or work-in-progress (WIP) code should be on `main`.
- **Feature Branches** – Every new feature or bug fix is developed on a separate branch created from `main`. Use descriptive names for branches (e.g., `feature/login-page` or `bugfix/api-timeout`) so their purpose is clear. This isolated workspace lets developers commit and experiment freely without affecting the stable code on `main`.
- **Pull Requests for Integration** – Changes from a feature branch are merged back into `main` via a **pull request (PR)**, never by direct pushes. Before merging, the team must review the PR and ensure all tests pass. This process provides code review, discussion, and quality checks before the code lands on `main`.
- **Clean Merges** – We prefer **squash merging** for PRs. Squash merging condenses all the commits from a feature branch into a single commit on `main`, resulting in a linear and clean commit history. After a successful merge, delete the feature branch to keep the repository tidy.

## Branch Protection Rules

To enforce the above workflow and protect the integrity of `main`, we configure **branch protection rules** on the `main` branch. These rules ensure that all changes go through proper review and CI checks before reaching `main`, and prevent risky actions like force pushes.

- **Require Pull Request Reviews** – At least **one approving code review** is required before a PR can be merged into `main`. The repository settings are configured to “Require a pull request before merging” and “Require approvals” (set to a minimum of 1 approval). This ensures every change on `main` has been reviewed for quality and correctness.
- **PR-Only Changes (No Direct Pushes)** – Developers **cannot push directly to `main`**. All changes must go through the PR process. The rule applies to all users, including administrators, to ensure that no one bypasses code review.
- **No Force Pushes or Deletions** – Force pushes to `main` are **blocked** and the branch cannot be deleted. Disallowing force pushes prevents history rewriting and accidental loss of commits, preserving the repository’s integrity.
- **Squash and Merge Only** – We encourage using **“Squash and Merge”** for all pull requests into `main`. This method creates a single commit per feature/bugfix, keeping the `main` history concise. Optionally, the repository can enforce a “Require linear history” rule to prevent merge commits entirely.
- **Status Checks and CI Enforcement** – Required status checks are enforced on `main`. This means a PR cannot be merged if designated CI checks (e.g., tests, builds) have failed or are missing. Additionally, we enable the “Require branches to be up to date before merging” rule, ensuring that feature branches are in sync with the latest `main` before merging.

## CI/CD Workflow

We use **GitHub Actions** for continuous integration (CI) and continuous deployment (CD) to Azure. Our CI/CD pipeline tests every change and deploys automatically when code is merged to `main`, enabling rapid and reliable releases.

- **Automated Testing (CI)** – Every pull request triggers a GitHub Actions workflow to run our test suite (as well as static analysis or build steps). The workflow is defined in a YAML file (e.g., `.github/workflows/ci.yml`) and runs on every push or PR. CI must pass before the PR can be merged, ensuring no failing code is introduced into `main`.
- **Continuous Deployment (CD) to Azure** – Once a PR is merged into `main`, a separate GitHub Actions workflow (e.g., `.github/workflows/deploy.yml`) automatically deploys the application to Azure. The process typically includes building the application, running final tests, and deploying to staging and/or production environments.
- **Staging and Production Environments** – Our deployment pipeline targets two environments:
    - **Staging** – Upon merging to `main`, the code is first deployed to a staging environment for validation. This environment mirrors production to perform smoke tests and QA verification.
    - **Production** – After successful staging validation (and optionally, a manual approval), the code is deployed to the production environment.
- **Required Checks for Deployment** – The deployment workflow depends on the success of the CI workflow. Only when all tests pass and all required status checks are met will the deployment proceed.
- **Secrets and Credentials** – GitHub Actions accesses Azure using securely stored secrets (e.g., `AZURE_CREDENTIALS`). These secrets are managed in the repository settings, ensuring that sensitive information is not exposed in the codebase.

## Best Practices for Collaboration

To maintain a reliable codebase and efficient workflow, we follow these best practices:

- **Descriptive Branch Names**  
  Use clear and consistent branch names that reflect the purpose of the branch. Examples:
    - `feature/login-page` – For a new login page feature.
    - `bugfix/api-timeout` – For addressing an API timeout issue.
    - `hotfix/payment-crash` – For an urgent fix on the payment module.

- **Small, Focused Pull Requests**  
  Keep PRs small and focused on a single concern. This approach makes reviews more manageable and reduces the likelihood of merge conflicts. Breaking down larger features into incremental PRs facilitates quicker feedback and smoother integration.

- **Clear Commit Messages**  
  Use consistent, informative commit messages that explain the what and why of changes.

- **Issue Tracking**  
  Utilize JIRA to track features, enhancements, and bugs. Ensure that every branch or PR is linked to an issue. This practice helps in understanding the context of changes and in maintaining a clear history of decisions.

- **Pull Request Template**  
  Provide a standardized PR template (**TBD**)
   - A consistent template (stored in `.github/PULL_REQUEST_TEMPLATE.md`) ensures that every PR contains the necessary information for reviewers.

- **Consistent Workflow and Communication**
    - Use PR comments for clarifications and suggestions.
    - Utilize the “Draft Pull Request” feature if the PR is not ready for full review.
    - Encourage timely reviews and constructive feedback.
    - Keep communication open on related issues through comments, and/or Slack channels.

  Maintaining a clear and consistent workflow ensures that development progresses smoothly and all team members are aligned.

By following this documentation, our team will maintain a robust development workflow that enforces code quality, integrates continuous delivery, and keeps our production environment stable. GitHub Flow, combined with strict branch protection and automated CI/CD deployments, supports rapid, reliable, and collaborative software development.
