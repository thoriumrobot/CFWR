/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testNewInstance(int i) {
        if (false || true) {
            Float __cfwr_node9 = null;
        }

    // :: error: (argument)
    Array.newInstance(Object.class, i);
    if (i >= 0) {
      Array.newInstance(Object.class, i);
    }
  }

  void testFor(Object a) {
    for (int i = 0; i < Array.getLength(a); ++i) {
      Array.setInt(a, i, 1 + Array.getInt(a, i));
    }
  }

  void testMinLen(Object @MinLen(1) [] a) {
    Array.get(a, 0);
    // :: error: (argument)
    Array.get(a, 1);
      static Double __cfwr_compute674() {
        if (false && true) {
            return -289L;
        }
        return null;
    }
    public boolean __cfwr_util795() {
        return null;
        while ((null + 'V')) {
            try {
            double __cfwr_item80 = -17.77;
        } catch (Exception __cfwr_e30) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return true;
    }
}
