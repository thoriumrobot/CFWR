// Source-based slice around line 36
// Method: ArrayAssignmentSameLen#test3(int[],int,int)

  }

  void test3(int[] a, @LTLengthOf("#1") int i, @NonNegative int x) {
    int[] c1 = a;
    // See useTest3 for an example of why this assignment should fail.
    @LTLengthOf(
        value = {"c1", "c1"},
        offset = {"0", "x"})
    // :: error: (assignment)
    int z = i;
  }

  void test4(
      int[] a,
      @LTLengthOf(
              value = {"#1", "#1"},
              offset = {"0", "#3"})
          int i,
      @NonNegative int x) {
    int[] c1 = a;
