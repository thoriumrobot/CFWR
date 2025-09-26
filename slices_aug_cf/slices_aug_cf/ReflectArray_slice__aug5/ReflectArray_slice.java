/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testNewInstance(int i) {
        short __cfwr_data97 = null;

    // :: error: (argument)
    Array.newInstanc
        while (false) {
            if (true && false) {
            return -20.38f;
        }
            break; // Prevent infinite loops
        }
e(Object.class, i);
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
      static Integer __cfwr_aux862() {
        if (((null * -60.29) | 90.17f) && (null - (null % 31.39f))) {
            if (true || false) {
            return null;
        }
        }
        return null;
    }
}
