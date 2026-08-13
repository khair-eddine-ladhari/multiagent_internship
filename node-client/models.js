/**
 * Data models shared on the Node.js side.
 *
 * These mirror python-server/models.py so the client and server
 * agree on shape. The Node client never computes these itself —
 * it just receives them as JSON from the Python server and can
 * wrap them in these classes for convenience/typing.
 */

class Product {
  constructor({
    id,
    name = "",
    price = null,
    source = "",
    url = "",
    specifications = {},
    extraInfo = {},
  }) {
    this.id = id ?? `prod_${crypto.randomUUID().slice(0, 8)}`;
    this.name = name;
    this.price = price;
    this.source = source;
    this.url = url;
    this.specifications = specifications;
    this.extraInfo = extraInfo;
  }

  summary() {
    const priceStr = this.price !== null ? `${this.price} TND` : "price unknown";
    return `${this.name} - ${priceStr} (${this.source})`;
  }
}

class SessionState {
  constructor({
    sessionId = crypto.randomUUID(),
    lastSearchResults = [],
    selectedProductId = null,
    lastQuery = null,
  } = {}) {
    this.sessionId = sessionId;
    this.lastSearchResults = lastSearchResults;
    this.selectedProductId = selectedProductId;
    this.lastQuery = lastQuery;
  }

  setResults(productIds) {
    this.lastSearchResults = productIds;
    this.selectedProductId = productIds.length > 0 ? productIds[0] : null;
  }

  select(productId) {
    if (this.lastSearchResults.includes(productId)) {
      this.selectedProductId = productId;
    }
  }
}

module.exports = { Product, SessionState };