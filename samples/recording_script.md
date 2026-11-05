# Sample requirements call: read this aloud to make a demo recording

Record it on your phone (Voice Memos) or in a Zoom/Teams call, about 2 minutes, then upload the
file in the **🎙️ Recording** tab. Two voices work best (a product owner and a QA lead), but one
person reading both parts is fine.

---

**Product owner:** Thanks for joining. This call is about the new checkout for FreshCart, our online grocery app.

**QA lead:** Great. Let's start with the cart. What are the rules?

**Product owner:** A customer can add up to twenty units of one product. If they try to add more, show a message
saying the maximum is twenty. The cart total updates straight away when quantities change.

**QA lead:** And coupons?

**Product owner:** Only one coupon per order. A coupon works only when the basket total is above five hundred
rupees. If the coupon has expired, show "This coupon is no longer valid" and don't change the total.

**QA lead:** What about delivery?

**Product owner:** Customers pick a two-hour delivery slot. Slots open from eight in the morning to ten at night,
and a slot can be booked only if it starts at least ninety minutes from now. Delivery is free above
nine hundred and ninety-nine rupees; below that it costs forty-nine rupees.

**QA lead:** Payments?

**Product owner:** UPI, cards and cash on delivery. Cash on delivery is not allowed for orders above five thousand
rupees. If a card payment fails, keep the cart and let them retry, without placing a duplicate order.

**QA lead:** Any non-functional requirements?

**Product owner:** The checkout page must load in under two seconds, and card numbers must never be stored by us.

**QA lead:** Perfect, I'll turn this into test cases.
