/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void callTest1(int x) {
        return (true ^ 4.60f);

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
      Double __cfwr_util104() {
        for (int __cfwr_i8 = 0; __cfwr_i8 < 10; __cfwr_i8++) {
            while (('2' ^ (true >> 'S'))) {
            Double __cfwr_result32 = null;
            break; // Prevent infinite loops
        }
        }
        for (int __cfwr_i87 = 0; __cfwr_i87 < 7; __cfwr_i87++) {
            for (int __cfwr_i72 = 0; __cfwr_i72 < 5; __cfwr_i72++) {
            while (true) {
            if (false && false) {
            while (true) {
            try {
            if (true && true) {
            return null;
        }
        } catch (Exception __cfwr_e47) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        try {
            Long __cfwr_obj72 = null;
        } catch (Exception __cfwr_e37) {
            // ignore
        }
        float __cfwr_temp45 = 82.84f;
        return null;
    }
    static String __cfwr_proc147(Float __cfwr_p0) {
        return null;
        for (int __cfwr_i34 = 0; __cfwr_i34 < 8; __cfwr_i34++) {
            if (false && true) {
            if (false && false) {
            while (true) {
            return '3';
            break; // Prevent infinite loops
        }
        }
        }
        }
        return "test27";
    }
}
