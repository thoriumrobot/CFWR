/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void callTest1(int x) {
        if (true && ('h' & true)) {
            Boolean __cfwr_result62 = null;
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
      protected char __cfwr_aux869(short __cfwr_p0) {
        Boolean __cfwr_node80 = null;
        return (false >> (-2.90f & null));
        Long __cfwr_val74 = null;
        if (('P' | (null / null)) && (null - -505)) {
            return 410L;
        }
        return '3';
    }
    char __cfwr_compute43(char __cfwr_p0, double __cfwr_p1, Boolean __cfwr_p2) {
        while (true) {
            boolean __cfwr_val9 = (543 & '2');
            break; // Prevent infinite loops
        }
        return '7';
    }
}
