/* Run with node tests/test_handcheck.js; no npm packages required. */
const {test} = require("node:test");
const assert = require("node:assert/strict");
const {emptyReview, validateSession} = require("../handcheck/gallery.js");

const sample = {sample_sha256: "fixed-sample", options: ["left", "right", "front", "behind"],
                items: [{id: "one"}, {id: "two"}]};
const review = {...emptyReview("one"), verdict: "reject", corrected_label: "right", notes: "Wrong relation"};
const session = {schema_version: 1, sample_sha256: "fixed-sample", reviewer: "Person C", reviews: [review]};

test("exported session round-trips through JSON without losing decisions or notes", () => {
  assert.deepEqual(validateSession(JSON.parse(JSON.stringify(session)), sample), session);
  assert.deepEqual(emptyReview("two"), {id: "two", verdict: "unreviewed", corrected_label: "", notes: ""});
});

test("changed sample and format are rejected", () => {
  assert.throws(() => validateSession({...session, sample_sha256: "changed"}, sample), /different sample/);
  assert.throws(() => validateSession({...session, schema_version: 2}, sample), /different sample/);
});

test("unknown and duplicate question IDs are rejected", () => {
  assert.throws(() => validateSession({...session, reviews: [review, review]}, sample));
  assert.throws(() => validateSession({...session, reviews: [{...review, id: "unknown"}]}, sample));
});

test("invalid decisions, corrections, and notes are rejected before restore", () => {
  for (const bad of [{...review, verdict: "maybe"}, {...review, corrected_label: "yes"},
                     {...review, verdict: "accept"}, {...review, notes: {html: "text"}}, null]) {
    assert.throws(() => validateSession({...session, reviews: [bad]}, sample));
  }
  assert.equal(session.reviews[0].verdict, "reject");
});

test("partial sessions and IDs resembling object properties remain valid", () => {
  assert.equal(validateSession({...session, reviews: []}, sample).reviews.length, 0);
  const unusual = {...sample, items: [{id: "__proto__"}]};
  assert.deepEqual(validateSession({...session, reviews: [emptyReview("__proto__")]}, unusual).reviews,
                   [emptyReview("__proto__")]);
});
