/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void callTest(int x) {
        boolean __cfwr_elem39 = (-902L << -873);

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
      public boolean __cfwr_process31(Float __cfwr_p0, Long __cfwr_p1) {
        for (int __cfwr_i98 = 0; __cfwr_i98 < 1; __cfwr_i98++) {
            try {
            for (int __cfwr_i8 = 0; __cfwr_i8 < 6; __cfwr_i8++) {
            if ((false * null) || ('W' ^ 51.33f)) {
            return 473L;
        }
        }
        } catch (Exception __cfwr_e65) {
            // ignore
        }
        }
        while ((null / false)) {
            double __cfwr_data68 = 57.23;
            break; // Prevent infinite loops
        }
        while ((null + -804L)) {
            for (int __cfwr_i6 = 0; __cfwr_i6 < 9; __cfwr_i6++) {
            for (int __cfwr_i32 = 0; __cfwr_i32 < 7; __cfwr_i32++) {
            for (int __cfwr_i44 = 0; __cfwr_i44 < 6; __cfwr_i44++) {
            Boolean __cfwr_obj95 = null;
        }
        }
        }
            break; // Prevent infinite loops
        }
        return 80.62f;
        return false;
    }
}
