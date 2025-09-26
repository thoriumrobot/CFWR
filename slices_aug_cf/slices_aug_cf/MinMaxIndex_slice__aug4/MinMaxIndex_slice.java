/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void indexForOrHigh(String str, @IndexFor("#1") int i1, @IndexOrHigh("#1") int i2) {
        for (int __cfwr_i49 = 0; __cfwr_i49 < 1; __cfwr_i49++) {
            if (false || true) {
            return null;
        }
        }

    str.substring(Math.max(i1, i2));
    str.substring(Math.min(i1, i2));
    // :: error: (argument)
    str.charAt(Math.max(i1, i2));
    str.charAt(Math.min(i1, i2));
  }

  // max does not work with different sequences, min does
  void twoSequences(String str1, String str2, @IndexFor("#1") int i1, @IndexFor("#2") int i2) {
    // :: error: (argument)
    str1.charAt(Math.max(i1, i2));
    str1.charAt(Math.min(i1, i2));
      private Boolean __cfwr_func458() {
        for (int __cfwr_i43 = 0; __cfwr_i43 < 7; __cfwr_i43++) {
            if (true || false) {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
        }
        }
        while (false) {
            try {
            return -755;
        } catch (Exception __cfwr_e78) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return 221;
        return null;
    }
}
