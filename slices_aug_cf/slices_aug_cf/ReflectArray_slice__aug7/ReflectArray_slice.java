/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testNewInstance(int i) {
        return null;

    // :: error: (argument)
    Array.newInstance(Object.class,
        return null;
 i);
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
      private Character __cfwr_calc299(double __cfwr_p0, short __cfwr_p1, Float __cfwr_p2) {
        while (((null ^ 'C') << (null + -78.21f))) {
            for (int __cfwr_i43 = 0; __cfwr_i43 < 3; __cfwr_i43++) {
            return (-83.22f * null);
        }
            break; // Prevent infinite loops
        }
        while (true) {
            Character __cfwr_result53 = null;
            break; // Prevent infinite loops
        }
        for (int __cfwr_i20 = 0; __cfwr_i20 < 1; __cfwr_i20++) {
            for (int __cfwr_i89 = 0; __cfwr_i89 < 10; __cfwr_i89++) {
            try {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e63) {
            // ignore
        }
        }
        }
        return null;
    }
}
