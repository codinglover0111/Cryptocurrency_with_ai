/** @type {import('eslint').Linter.Config} */
module.exports = {
  root: true,
  extends: ["next", "next/core-web-vitals"],
  parserOptions: {
    project: true
  },
  rules: {
    "@next/next/no-img-element": "off"
  }
};
