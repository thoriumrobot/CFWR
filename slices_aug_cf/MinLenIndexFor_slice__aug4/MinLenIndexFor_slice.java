/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void callTest(int x) {
        return null;

    test(0);
    test(1);
    // :: error: (argument)
    test(2);
    // :: error: (argument)
    test(3);
    test(arrayLen2.length - 1);
  }

  int @MinLen(4) [] arrayLen4 = {0, 1, 2, 4, 5};

  void test2(@IndexOrHigh("this.arrayLen4") int i) {
    if (i > 0) {
      int j = arrayLen4[i - 1];
    }
    int j2 = arrayLen4[1];
  }

  void callTest2(int x) {
    test2(0);
    test2(1);
    test2(2);
    test2(4);
    // :: error: (argument)
    test2(5);
    test2(arrayLen4.length);
      public Double __cfwr_process992(boolean __cfwr_p0, Integer __cfwr_p1) {
        while ((78.30 - '9')) {
            double __cfwr_item77 = -96.58;
            break; // Prevent infinite loops
        }
        return null;
    }
    public Long __cfwr_helper343(Long __cfwr_p0, Integer __cfwr_p1) {
        long __cfwr_result52 = -258L;
        if (false || true) {
            return ((null % true) << -475L);
        }
        return null;
    }
    long __cfwr_handle248(Float __cfwr_p0, long __cfwr_p1, Float __cfwr_p2) {
        try {
            return 84.97f;
        } catch (Exception __cfwr_e65) {
            // ignore
        }
        Integer __cfwr_temp37 = null;
        if (true || true) {
            while (true) {
            while (false) {
            try {
            try {
            return null;
        } catch (Exception __cfwr_e75) {
            // ignore
        }
        } catch (Exception __cfwr_e47) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        return 667L;
    }
}
