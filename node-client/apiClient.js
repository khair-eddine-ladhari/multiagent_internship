/**
 * apiClient.js
 *
 * Node.js side of the app. Does NOT run any search, agent, or database
 * logic itself — it just calls the Python server, which owns the flow.
 *
 * Responsibilities:
 *  - send the user's message + current session state to the Python server
 *  - receive back { report, sessionState } and hand it to the caller
 *  - keep sessionState in memory between calls (per user/session)
 */

require("dotenv").config();

const { SessionState } = require("./models");

const PYTHON_SERVER_URL = process.env.PYTHON_SERVER_URL || "http://localhost:8000";

/**
 * Send a user message to the Python flow endpoint.
 *
 * @param {string} message - the raw user message
 * @param {SessionState} sessionState - current session state (memory)
 * @returns {Promise<{ report: string, sessionState: SessionState }>}
 */
async function askAssistant(message, sessionState) {
  const response = await fetch(`${PYTHON_SERVER_URL}/flow`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      session_state: {
        session_id: sessionState.sessionId,
        last_search_results: sessionState.lastSearchResults,
        selected_product_id: sessionState.selectedProductId,
        last_query: sessionState.lastQuery,
      },
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Python server error (${response.status}): ${errorText}`);
  }

  const data = await response.json();

  // Python server returns the updated session state as plain JSON.
  // Rebuild it into a SessionState instance so the caller keeps using
  // the same class/methods (select, setResults, etc.) on the next turn.
  const updatedState = new SessionState({
    sessionId: data.session_state.session_id,
    lastSearchResults: data.session_state.last_search_results,
    selectedProductId: data.session_state.selected_product_id,
    lastQuery: data.session_state.last_query,
  });

  return {
    report: data.report,
    sessionState: updatedState,
  };
}

module.exports = { askAssistant };

// --- Example usage ---
// (async () => {
//   let session = new SessionState();
//   const first = await askAssistant("Find me laptops under 3000 TND", session);
//   console.log(first.report);
//   session = first.sessionState;
//
//   const followUp = await askAssistant("Tell me more about the second one", session);
//   console.log(followUp.report);
// })();