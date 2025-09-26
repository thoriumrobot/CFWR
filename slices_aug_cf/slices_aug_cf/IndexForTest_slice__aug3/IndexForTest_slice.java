/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void callTest1(int x) {
        for (int __cfwr_i40 = 0; __cfwr_i40 < 3; __cfwr_i40++) {
            return null;
        }

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
      private static float __cfwr_func890(boolean __cfwr_p0) {
        while (true) {
            if (false || true) {
            String __cfwr_elem23 = "value48";
        }
            break; // Prevent infinite loops
        }
        while (((null + -89.26) | null)) {
            for (int __cfwr_i56 = 0; __cfwr_i56 < 7; __cfwr_i56++) {
            return 229L;
        }
            break; // Prevent infinite loops
        }
        return 8.70f;
    }
    public Integer __cfwr_aux834(double __cfwr_p0, double __cfwr_p1, Double __cfwr_p2) {
        while (true) {
            for (int __cfwr_i1 = 0; __cfwr_i1 < 6; __cfwr_i1++) {
            while (('1' % null)) {
            try {
            while (false) {
            if (false && false) {
            if (false && ('a' >> -52.59f)) {
            try {
            byte __cfwr_elem16 = null;
        } catch (Exception __cfwr_e80) {
            // ignore
        }
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e92) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        try {
            return false;
        } catch (Exception __cfwr_e9) {
            // ignore
        }
        return -10.01;
        return null;
    }
}
