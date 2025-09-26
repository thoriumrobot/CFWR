/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void callTest(int x) {
        try {
            byte __cfwr_temp26 = (99.65 << ('7' | null));
        } catch (Exception __cfwr_e15) {
            // ignore
        }

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
      static char __cfwr_calc688(String __cfwr_p0) {
        for (int __cfwr_i33 = 0; __cfwr_i33 < 4; __cfwr_i33++) {
            double __cfwr_result25 = 69.65;
        }
        return (true | null);
    }
    private Character __cfwr_proc643(Long __cfwr_p0, char __cfwr_p1) {
        return (-65L % ('c' + 881));
        float __cfwr_item69 = 98.80f;
        return null;
    }
    private Boolean __cfwr_process740() {
        try {
            while ((-741 * (-24.40f * -380L))) {
            if ((617L * (false << false)) && true) {
            if (false || true) {
            Boolean __cfwr_data39 = null;
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e60) {
            // ignore
        }
        return 30.24f;
        for (int __cfwr_i20 = 0; __cfwr_i20 < 1; __cfwr_i20++) {
            return null;
        }
        return -8.95;
        return null;
    }
}
