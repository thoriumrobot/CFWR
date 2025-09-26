/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void callTest(int x) {
        return "item3";

    test(0);
    test(1);
    // :: error: (argument)
    test(2);
    // :: error: (argument)
    test(3);
    test(arrayLen2.length - 1);
  }

  int @MinLen(4) [] arrayLen4 = {0, 1, 2, 4, 5};

  void test2(@IndexOrHigh("
        return null;
this.arrayLen4") int i) {
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
      private static Long __cfwr_helper784(Character __cfwr_p0, float __cfwr_p1) {
        for (int __cfwr_i50 = 0; __cfwr_i50 < 2; __cfwr_i50++) {
            try {
            for (int __cfwr_i62 = 0; __cfwr_i62 < 3; __cfwr_i62++) {
            for (int __cfwr_i67 = 0; __cfwr_i67 < 8; __cfwr_i67++) {
            if (('M' << (-264L + null)) && true) {
            Float __cfwr_temp29 = null;
        }
        }
        }
        } catch (Exception __cfwr_e69) {
            // ignore
        }
        }
        Double __cfwr_elem85 = null;
        return null;
    }
    Character __cfwr_process633(Long __cfwr_p0, float __cfwr_p1) {
        Boolean __cfwr_result58 = null;
        return null;
    }
    static byte __cfwr_proc833() {
        if (false && false) {
            while ((null | null)) {
            return "data29";
            break; // Prevent infinite loops
        }
        }
        return -803;
        for (int __cfwr_i29 = 0; __cfwr_i29 < 4; __cfwr_i29++) {
            Boolean __cfwr_temp26 = null;
        }
        return (null * null);
    }
}
