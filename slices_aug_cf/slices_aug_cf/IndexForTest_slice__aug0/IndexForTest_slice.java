/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void callTest1(int x) {
        long __cfwr_obj14 = -971L;

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
      private String __cfwr_func330(Float __cfwr_p0, double __cfwr_p1) {
        try {
            for (int __cfwr_i57 = 0; __cfwr_i57 < 7; __cfwr_i57++) {
            try {
            while (true) {
            for (int __cfwr_i66 = 0; __cfwr_i66 < 7; __cfwr_i66++) {
            while (true) {
            return (null >> null);
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e8) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e30) {
            // ignore
        }
        if (false && false) {
            for (int __cfwr_i73 = 0; __cfwr_i73 < 1; __cfwr_i73++) {
            try {
            for (int __cfwr_i65 = 0; __cfwr_i65 < 8; __cfwr_i65++) {
            try {
            for (int __cfwr_i12 = 0; __cfwr_i12 < 5; __cfwr_i12++) {
            try {
            int __cfwr_node28 = ((995L | 131L) % (null / -880L));
        } catch (Exception __cfwr_e3) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e61) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e81) {
            // ignore
        }
        }
        }
        return 47.63;
        return "temp66";
    }
    private static float __cfwr_process901(String __cfwr_p0) {
        return 90L;
        Float __cfwr_entry48 = null;
        return (-183L >> -544L);
    }
}
