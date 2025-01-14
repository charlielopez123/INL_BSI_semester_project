#################
having .rst fun ?
#################

Look how nice rst can get!::

   Long live documentation!

.. warning:: Yes, you can use all of them into your code docstrings!

.. contents::
  :local:
  :depth: 1

***************
Heading Levels
***************

:: 

   #############
   Huge
   #############

   *************
   Very big
   *************

   ===========
   Big
   ===========

   Not that big
   ************

   A bit small
   ===========

   I cannot see it!
   ~~~~~~~~~~~~~~~~~

************************************************
Paragraph Text and Commented Text
************************************************

Let's write a *very* **very** important paragraph.

| I learnt this little  
| trick by adding ``|`` at the 
| beginning of the line.

.. let's comment about this. Do you agree ? or TODO ?


*****************
Code Formatting
*****************

What if I want ``some inline code`` ?

::

 I wanna say something in code-style

.. code-block:: python

   # no, let's say it in python
   def foo():
      return 'bar'

.. code-block:: bash

   # or maybe bash
   echo "Hello, world!"

*****************
Lists
*****************

#. this is the first item
#. this is the second item
#. guess this one ?

* ha, who am I now ?
* don't know my number ?
* yeah I know
* boring, isn't it ?

* I see what
   * you did there
   * and there
* and there
   * not funny



*********************************
Notes and Warnings
*********************************

.. note::
   This is note text.

   This is still a note


.. warning::
   Warnings are formatted in the same way as notes. In the same way, lines must
   be broken and indented under the warning tag.


.. _Lenna image:

.. figure:: https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png
   :width: 256
   
   Recognize me ?


****************************
Cross-References
****************************

Did you recognize her ? :ref:`Lenna image`  
Or do you prefer :ref:`Lenna only<Lenna image>` ?

We're using autosection label which is pretty cool because you can
ref any section in any file just by given the section title, such
as :ref:`api:API Reference`. You can of course give it a
:ref:`another name<api:API Reference>` if you want.


************************************
Tables
************************************

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - First Name
     - Last Name
     - Residence
   * - Look
     - at
     - this!
   * - isn't
     - it
     - amazing?

.. list-table::
   :widths: 15 15 70
   :stub-columns: 1

   * - First Name
     - I
     - prefer
   * - Last Name
     - to
     - read
   * - Residence
     - like
     - this
