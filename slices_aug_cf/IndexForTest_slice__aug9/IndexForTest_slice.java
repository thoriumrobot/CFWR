/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void callTest1(int x) {
        while (true) {
            Long __cfwr_node40 = null;
            break; // Prevent infinite loops
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
      public Boolean __cfwr_func204(int __cfwr_p0) {
        if (true && false) {
            try {
            return null;
        } catch (Exception __cfwr_e69) {
            // ignore
        }
        }
        return null;
        if (true || (-304L * -9.57f)) {
            while (true) {
            return "item79";
            break; // Prevent infinite loops
        }
        }
        Double __cfwr_elem33 = null;
        return null;
    }
}
