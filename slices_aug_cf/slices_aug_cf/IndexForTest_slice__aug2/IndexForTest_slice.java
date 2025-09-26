/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void callTest1(int x) {
        long __cfwr_entry76 = 112L;

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
      private Integer __cfwr_util44(Double __cfwr_p0, boolean __cfwr_p1) {
        for (int __cfwr_i85 = 0; __cfwr_i85 < 5; __cfwr_i85++) {
            try {
            Character __cfwr_node6 = null;
        } catch (Exception __cfwr_e65) {
            // ignore
        }
        }
        try {
            for (int __cfwr_i71 = 0; __cfwr_i71 < 1; __cfwr_i71++) {
            String __cfwr_val65 = "world18";
        }
        } catch (Exception __cfwr_e86) {
            // ignore
        }
        for (int __cfwr_i2 = 0; __cfwr_i2 < 4; __cfwr_i2++) {
            while (false) {
            Long __cfwr_node6 = null;
            break; // Prevent infinite loops
        }
        }
        return null;
    }
    protected Integer __cfwr_func652() {
        try {
            byte __cfwr_result63 = null;
        } catch (Exception __cfwr_e73) {
            // ignore
        }
        for (int __cfwr_i61 = 0; __cfwr_i61 < 8; __cfwr_i61++) {
            while ((81.59f | ('D' % -508L))) {
            Object __cfwr_entry31 = null;
            break; // Prevent infinite loops
        }
        }
        for (int __cfwr_i25 = 0; __cfwr_i25 < 2; __cfwr_i25++) {
            String __cfwr_data18 = "data15";
        }
        return null;
    }
    public static Double __cfwr_proc877(Integer __cfwr_p0) {
        Integer __cfwr_obj84 = null;
        while ((null ^ 75.91f)) {
            try {
            if (true && true) {
            try {
            return null;
        } catch (Exception __cfwr_e28) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e72) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        if (false || (232L - (null << 'D'))) {
            while (false) {
            for (int __cfwr_i50 = 0; __cfwr_i50 < 3; __cfwr_i50++) {
            return 324;
        }
            break; // Prevent infinite loops
        }
        }
        return null;
    }
}
