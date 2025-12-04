The right historical window depends on **your seasonality, product behavior, model class, and business needs**, but here’s a principled recommendation based on your case (daily data, mixed stable and seasonal products).

---

# ✅ **Short Answer (Practical Recommendation)**

**Keep *at least* 2 years of daily history**, **preferably 3 years** if available.

For your current dataset (≈1100 rows ≈ 3 years):
👉 **You are in a very good range already.**

---

# 🧠 **Why 2–3 Years Is the Sweet Spot**

## **1. You need at least 1 full seasonal cycle**

If your products have yearly seasonality (e.g., spikes in summer/winter), having a full year is **the minimum**.

But *one* seasonal cycle is rarely enough for models to be robust.

---

## **2. Two cycles lets you detect changes**

With 2 years:

* You can see if the seasonal pattern is consistent
* You can detect structural changes (e.g., pricing policy change, new competition)

---

## **3. Three cycles makes seasonality modeling much more reliable**

Particularly for:

* **Prophet** (multi-year seasonality modeling)
* **SARIMA** (needs repeated seasonal structure)
* **Machine learning models** (need enough training examples across different regimes)

If your products show intermittent demand, longer history smooths volatility.

---

# 📈 **Should you go further than 3 years?**

### **You *can*, but it's not always beneficial.**

Use >3 years when:

* Product behavior is stable long-term
* Price changes do not alter demand structure
* Category does not undergo major lifecycle transitions
* You want robust long-horizon forecasting

Use ≤3 years when:

* Products frequently change price strategy
* Demand shifts strongly due to trends (fashions, electronics, etc.)
* Older data becomes unrepresentative
* There were major catalog or store changes

In practice, for most retail forecasting problems:

> **3 years = best tradeoff between signal and relevance.**

---

# 🔍 **In your specific case (current: ~1100 daily rows):**

01-01-2023 → today gives you:

* 2023
* 2024
* 2025 (partial)

That’s *almost 3 full years*.

✔️ More than enough for:

* Seasonal detection
* Reliable trend modeling
* ML-based forecasting

Only add more history if:

### **1. Products have *very sharp* seasonal cycles**

(e.g., sold only in December)

### **2. You want to model multi-year structural trends**

Optional, depends on business.

### **3. Business objects have long lifecycles**

(e.g., industrial equipment)

### **4. Older data is clean and comparable**

(Same product ID semantics, same pricing structure)

---

# ⚠️ When NOT to use older historical data

Avoid older data if:

* Product IDs changed or were remapped
* Sales distribution channels changed
* Products were rebranded or reconfigured
* COVID- or anomaly-period behavior makes the series misleading

(Old anomalies can harm machine learning models more than help them.)

---

# 🛠️ **Practical heuristic to automate this**

For each product:

1. Compute **stability score**

   * CV of sales
   * Price volatility
   * Missing data frequency
2. If stable → use full history (up to 3 years or more)
3. If highly volatile or recency-weighted → use the most recent 18–24 months
4. If seasonal but intermittent → use minimum 24 months

I can help you write a function that automatically decides how much history to use per product.

---

# ✔️ Final Recommendation

Since you currently pull ~1100 rows (~3 years):

> **Keep what you have. Do not truncate.
> If more data before 2023 exists, only include it if product definitions and business conditions remained stable.**

---

# If you want next steps…

I can help you:

* Build a **history window selector** function per SKU
* Build the next step of your architecture (feature engineering)
* Add profiling metrics to evaluate the optimal lookback period

Would you like to continue with the **feature engineering plan** or refine the **data profiling module** further?
