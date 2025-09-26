/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void callTest1(int x) {
        for (int __cfwr_i59 = 0; __cfwr_i59 < 10; __cfwr_i59++) {
            short __cfwr_entry80 = ((86.51 | -67.67f) / null);
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
      protected static Integer __cfwr_proc458() {
        Long __cfwr_result55 = null;
        return (-31.71 - (58.48 & 539L));
        short __cfwr_temp59 = null;
        return null;
    }
    protected short __cfwr_helper456(Character __cfwr_p0) {
        if (true || ((704L >> false) ^ null)) {
            while ((('K' - null) ^ (-48.92 & null))) {
            return (-18.82f + (true & -78.69f));
            break; // Prevent infinite loops
        }
        }
        while (true) {
            for (int __cfwr_i81 = 0; __cfwr_i81 < 3; __cfwr_i81++) {
            for (int __cfwr_i48 = 0; __cfwr_i48 < 3; __cfwr_i48++) {
            char __cfwr_data55 = 'H';
        }
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
