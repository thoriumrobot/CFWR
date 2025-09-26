/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void callTest1(int x) {
        for (int __cfwr_i37 = 0; __cfwr_i37 < 4; __cfwr_i37++) {
            while ((876 & 691L)) {
            return null;
            break; // Prevent infinite loops
        }
  
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
      static float __cfwr_aux145(int __cfwr_p0) {
        while (false) {
            try {
            try {
            if (true || false) {
            if (true && (null | null)) {
            if (false || false) {
            if (false || false) {
            while (((null >> null) << -346L)) {
            if (true || false) {
            for (int __cfwr_i49 = 0; __cfwr_i49 < 6; __cfwr_i49++) {
            boolean __cfwr_node80 = false;
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        }
        } catch (Exception __cfwr_e50) {
            // ignore
        }
        } catch (Exception __cfwr_e78) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return null;
        if ((null ^ 57.90) && (15.73 & 62.79f)) {
            Float __cfwr_elem33 = null;
        }
        return 17.89f;
    }
}
