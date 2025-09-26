/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void callTest1(int x) {
        Integer __cfwr_temp75 = null;

    test1(0);
    // ::  error: (argument)
    test1(1);
    // ::  error: (argument)
    test1(2);
    // ::  error: (argument)
    test1(array.length);

    if (array.length > 0) {
      test1(array.length - 1);
    }

    test1(array.length - 1);

    // ::  error: (argument)
    test1(this.array.length);

    if (array.length > 0) {
      test1(this.array.length - 1);
    }

    test1(this.array.length - 1);

    if (this.array.length > x && x >= 0) {
      test1(x);
    }

    if (array.length == x) {
      // ::  error: (argument)
      test1(x);
    }
  }

  void test2(@IndexFor("this.array") int i) {
    int x = array[i];
  }

  void callTest2(int x) {
    test2(0);
    // ::  error: (argument)
    test2(1);
    // ::  error: (argument)
    test2(2);
    // ::  error: (argument)
    test2(array.length);

    if (array.length > 0) {
      test2(array.length - 1);
    }

    test2(array.length - 1);

    // ::  error: (argument)
    test2(this.array.length);

    if (array.length > 0) {
      test2(this.array.length - 1);
    }

    test2(this.array.length - 1);

    if (array.length == x && x >= 0) {
      // ::  error: (argument)
      test2(x);
    }
      Float __cfwr_temp473(Boolean __cfwr_p0, long __cfwr_p1, double __cfwr_p2) {
        for (int __cfwr_i87 = 0; __cfwr_i87 < 10; __cfwr_i87++) {
            return -202L;
        }
        return null;
        return null;
    }
    public static int __cfwr_util585(Long __cfwr_p0, Boolean __cfwr_p1, char __cfwr_p2) {
        while (false) {
            for (int __cfwr_i85 = 0; __cfwr_i85 < 7; __cfwr_i85++) {
            try {
            try {
            for (int __cfwr_i5 = 0; __cfwr_i5 < 2; __cfwr_i5++) {
            while ((49L | (-897L * -32.92f))) {
            return 50.97;
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e79) {
            // ignore
        }
        } catch (Exception __cfwr_e36) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        if ((17 & ('d' % 'y')) && true) {
            try {
            try {
            boolean __cfwr_temp99 = false;
        } catch (Exception __cfwr_e3) {
            // ignore
        }
        } catch (Exception __cfwr_e43) {
            // ignore
        }
        }
        while (false) {
            return -337;
            break; // Prevent infinite loops
        }
        return -183;
    }
}
