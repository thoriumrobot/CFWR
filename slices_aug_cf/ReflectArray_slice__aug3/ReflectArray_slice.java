/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testNewInstance(int i) {
        String __cfwr_elem66 = "hello74";

    // :: error: (argument)
    Array.newInstance(Object.class, i);
    if (i >= 0) {
      Array.newInstance(Object.class, i);
    }
  }

  void testFor(Object a) {
    for (int i = 0; i < Array.getLength(a); ++i) {
      Array.setInt(a, i, 1 + Array.getInt(a, i));
    }
  }

  void testMinLen(Object @MinLen(1) [] a) {
    Array.get(a, 0);
    // :: error: (argument)
    Array.get(a, 1);
      private static Long __cfwr_aux266() {
        long __cfwr_entry14 = (false >> true);
        return null;
    }
}
